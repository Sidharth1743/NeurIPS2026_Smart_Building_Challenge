/* Complete checkpoint for the r116-style single-GPU asynchronous trainer.
 * Captures the already-prefetched rollout and the state after logging. */
#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <spawn.h>
#include <errno.h>

extern char** environ;
static int g_full_checkpoint = 0;
static int g_partial_resume = 0;
static int g_portable_resume = 0;
static int g_full_restored = 0;

// Latest means the highest numbered committed snapshot in this run, never a
// newer incomplete .bin or a different run selected by file ctime.
static const char* sb_full_latest(const char* dir, char* out, size_t size) {
    DIR* dp=opendir(dir); if(!dp)sbfull::fail("cannot open checkpoint directory",dir);
    char best[32]={}; struct dirent* ent;
    while((ent=readdir(dp))) {
        const char* n=ent->d_name;
        if(strlen(n)!=20 || strcmp(n+16,".bin"))continue;
        bool numeric=true; for(int i=0;i<16;++i)if(n[i]<'0'||n[i]>'9')numeric=false;
        if(!numeric || strcmp(n,best)<=0)continue;
        char full[4200];
        if(snprintf(full,sizeof(full),"%s/%s.full",dir,n)>=(int)sizeof(full))sbfull::fail("path too long");
        struct stat st{}; if(stat(full,&st)==0 && S_ISREG(st.st_mode))snprintf(best,sizeof(best),"%s",n);
    }
    closedir(dp); if(!best[0])sbfull::fail("no committed complete checkpoint",dir);
    if(snprintf(out,size,"%s/%s",dir,best)>=(int)size)sbfull::fail("path too long");
    return out;
}

// Keep ownership of a complete-checkpoint directory for the life of the
// process. The lock is released by the OS even if the process is killed.
static void sb_full_lock_directory(const char* dir) {
    char path[4200];
    if (snprintf(path,sizeof(path),"%s/.full-checkpoint.lock",dir)>=(int)sizeof(path))
        sbfull::fail("lock path too long");
    int fd=open(path,O_CREAT|O_RDWR,0600);
    if(fd<0 || flock(fd,LOCK_EX|LOCK_NB))sbfull::fail("checkpoint directory is locked",path);
}

// A crash may leave files for the next step before the .full commit marker.
// Preserve those files under unique names before replaying that step.
static void sb_full_prepare_write(const char* ckpt) {
    char committed[4200];
    if(snprintf(committed,sizeof(committed),"%s.full",ckpt)>=(int)sizeof(committed))
        sbfull::fail("checkpoint path too long");
    if(access(committed,F_OK)==0)sbfull::fail("refusing to overwrite committed snapshot",committed);
    const char* suffixes[]={"", ".mb", ".hadam"};
    for(const char* suffix:suffixes) {
        char source[4200], preserved[4260];
        if(snprintf(source,sizeof(source),"%s%s",ckpt,suffix)>=(int)sizeof(source))
            sbfull::fail("checkpoint path too long");
        if(access(source,F_OK)!=0) { if(errno!=ENOENT)sbfull::fail("cannot inspect checkpoint",source); continue; }
        if(!g_full_restored)sbfull::fail("refusing to overwrite existing checkpoint",source);
        bool linked=false;
        for(unsigned serial=0;serial<1000000;++serial) {
            if(snprintf(preserved,sizeof(preserved),"%s.orphan.%ld.%u",source,(long)getpid(),serial)>=(int)sizeof(preserved))
                sbfull::fail("checkpoint path too long");
            if(!link(source,preserved)) { linked=true; break; }
            if(errno!=EEXIST)sbfull::fail("cannot preserve incomplete checkpoint",source);
        }
        if(!linked||unlink(source))sbfull::fail("cannot preserve incomplete checkpoint",source);
        fprintf(stderr,"FULL_CHECKPOINT preserved incomplete component %s as %s\n",source,preserved);
    }
}

static void sb_full_sync_file(const char* path) {
    int fd=open(path,O_RDONLY);
    if(fd<0 || fsync(fd) || close(fd))sbfull::fail("cannot sync checkpoint component",path);
}

struct SbFullHeader {
    char magic[8]; uint32_t version, header_bytes;
    char executable_sha256[65]; unsigned char gpu_uuid[16];
    char driver_release[96];
    int32_t driver_version, runtime_version, cublas_version;
    uint64_t config_hash, weights_hash, momentum_hash, checksum;
    int64_t epoch, global_step, total_steps;
    int32_t ready_slot, write_slot, write_current, boot;
    int32_t hadam_len; int64_t hadam_off;
    double uptime, since_log; int64_t last_log_step;
};

static void sb_full_exe_sha(char out[65]) {
    char path[80]; snprintf(path,sizeof(path),"/proc/%ld/exe",(long)getpid());
    int fd[2]; if (pipe(fd)) sbfull::fail("pipe failed");
    posix_spawn_file_actions_t a; posix_spawn_file_actions_init(&a);
    posix_spawn_file_actions_adddup2(&a,fd[1],STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&a,fd[0]); posix_spawn_file_actions_addclose(&a,fd[1]);
    char program[]="sha256sum"; char* argv[]={program,path,nullptr}; pid_t pid;
    int rc=posix_spawnp(&pid,program,&a,nullptr,argv,environ);
    posix_spawn_file_actions_destroy(&a); close(fd[1]);
    if (rc) sbfull::fail("sha256sum unavailable");
    char line[256]={}; size_t n=0;
    while (n<sizeof(line)-1) { ssize_t r=read(fd[0],line+n,sizeof(line)-1-n); if (r<=0) break; n+=(size_t)r; }
    close(fd[0]); int status=0;
    if (waitpid(pid,&status,0)!=pid || !WIFEXITED(status) || WEXITSTATUS(status) || n<64)
        sbfull::fail("cannot hash executable");
    for (int i=0;i<64;++i) if (!((line[i]>='0'&&line[i]<='9')||(line[i]>='a'&&line[i]<='f')))
        sbfull::fail("invalid executable digest");
    memcpy(out,line,64); out[64]=0;
}
static uint64_t sb_full_file_hash(const char* path) {
    FILE* f=fopen(path,"rb"); if(!f) sbfull::fail("missing checkpoint component",path);
    std::vector<unsigned char> b(4*1024*1024); uint64_t h=sbfull::FNV_OFFSET;
    while(true){size_t n=fread(b.data(),1,b.size(),f);h=sbfull::hash_add(h,b.data(),n);
        if(n<b.size()){if(ferror(f))sbfull::fail("read failed",path);break;}}
    if(fclose(f))sbfull::fail("close failed",path);return h;
}
static uint64_t sb_full_config_hash(Ini* ini) {
    std::vector<std::string> v;
    for(int s=0;s<ini->num_sections;++s){
        Dict& d=ini->sections[s];
        for(int i=0;i<d.size;++i){
            DictItem& x=d.items[i];
            if(!strcmp(d.name,"base") && (!strcmp(x.key,"load_model_path") ||
               !strcmp(x.key,"checkpoint_dir") || !strcmp(x.key,"log_dir") || !strcmp(x.key,"run_id") ||
               !strcmp(x.key,"portable_resume"))) continue;
            std::string z=std::string(d.name)+"."+x.key+"=";
            if(x.str) z+=x.str; else { char b[80];snprintf(b,sizeof(b),"%.17g",x.value);z+=b; }
            v.push_back(z);
        }
    }
    for(char** e=environ;e&&*e;++e) {
        if(strncmp(*e,"SB_",3)==0 && strncmp(*e,"SB_SNAPSHOT=",12)!=0 &&
           strncmp(*e,"SB_SNAP_ENV=",12)!=0 && strncmp(*e,"SB_SNAP_EVERY=",14)!=0 &&
           strncmp(*e,"SB_GPU=",7)!=0) v.emplace_back(*e);
        else if(g_portable_resume && (!strncmp(*e,"CUDA_VISIBLE_DEVICES=",21) || !strncmp(*e,"CUDA_DEVICE_ORDER=",18))) continue;
        else if(strncmp(*e,"CUDA_",5)==0 || strncmp(*e,"CUBLAS_",7)==0 ||
                strncmp(*e,"NVIDIA_TF32_OVERRIDE=",21)==0) v.emplace_back(*e);
    }
    std::sort(v.begin(),v.end()); uint64_t h=sbfull::FNV_OFFSET;
    for(const auto& x:v){h=sbfull::hash_add(h,x.data(),x.size());char z=0;h=sbfull::hash_add(h,&z,1);}
    return h;
}
static void sb_full_platform(SbFullHeader& h) {
    sb_full_exe_sha(h.executable_sha256);
    int dev=0; sbfull::ck(cudaGetDevice(&dev)); cudaDeviceProp prop{};
    sbfull::ck(cudaGetDeviceProperties(&prop,dev)); memcpy(h.gpu_uuid,&prop.uuid,16);
    sbfull::ck(cudaDriverGetVersion(&h.driver_version)); sbfull::ck(cudaRuntimeGetVersion(&h.runtime_version));
    if(nvmlSystemGetDriverVersion(h.driver_release,sizeof(h.driver_release))!=NVML_SUCCESS)
        sbfull::fail("NVIDIA driver release unavailable");
    cublasHandle_t cb; if(cublasCreate(&cb)!=CUBLAS_STATUS_SUCCESS)sbfull::fail("cublasCreate failed");
    if(cublasGetVersion(cb,&h.cublas_version)!=CUBLAS_STATUS_SUCCESS)sbfull::fail("cublas version unavailable");
    if(cublasDestroy(cb)!=CUBLAS_STATUS_SUCCESS)sbfull::fail("cublasDestroy failed");
}
static bool sb_full_supported(PuffeRL* p) {
    return PUF_BACKEND==PUF_GPU && !USE_BF16 && p->hypers.world_size==1 && p->num_policies==1 &&
        p->hypers.num_buffers==1 && p->hypers.async && p->hypers.cudagraphs==1 && !p->hypers.reset_every_horizon &&
        g_checkpoint_optim && g_sbs_zone_mlp && g_sbs_zone_critic && g_sbs_value_shared && g_sbs_role_adam &&
        g_sbs_valve_zone && g_headwise_ratio && g_beta && g_beta_cn_tau>0 && !g_beta_mean_precision &&
        !g_sbs_zone_value && !g_sbs_room_value && !g_sbs_components && !g_sbs_ahu_critic &&
        !g_sbs_zone_value_residual && !g_sbs_ahu_mlp && !g_zone_pass_refresh && !g_sbs_team &&
        !g_sbs_idbd && !g_sbs_value_adam && !g_sbs_vsg && !g_sbs_vtc && !g_valve_global &&
        !sbgpu::g_zone_night_share && !g_adam && !g_cb_enabled && !g_entropy_current && !g_ent_target_on && !g_ent_role_on &&
        !g_shuffle && !g_freeze_return && g_action_repeat==1 && !g_ars && !getenv("SB_RECORD") &&
        sbgpu::sb_full_env_supported();
}
struct SbLogItem { char key[PUF_DICT_MAX_KEY]; double value; };
static void sb_full_dict(sbfull::Stream& io, Dict* d) {
    int n=d->size; io.host("log.count",&n);
    if(n<0 || n>100000)sbfull::fail("invalid log size");
    std::vector<SbLogItem> items((size_t)n);
    if(!io.reading)for(int i=0;i<n;++i){
        if(d->items[i].str || d->items[i].values)sbfull::fail("non-numeric log item");
        snprintf(items[i].key,sizeof(items[i].key),"%s",d->items[i].key);items[i].value=d->items[i].value;
    }
    io.host("log.items",items.data(),items.size());
    if(io.reading){dict_clear(d);for(auto& x:items){
        if(!memchr(x.key,0,sizeof(x.key)))sbfull::fail("invalid log key");dict_set(d,x.key,x.value);
    }}
}
static void sb_full_payload(sbfull::Stream& io, PuffeRL* p, SbFullHeader& hdr,
                            Dict* last, PufLogHistory* history, TrainResult* result) {
    io.block("trainer.params",p->policies[0].params_alloc.mem,p->policies[0].params_alloc.total_bytes,true);
    io.block("trainer.activations",p->activ_alloc.mem,p->activ_alloc.total_bytes,true);
    io.block("trainer.gradients",p->grads_alloc.mem,p->grads_alloc.total_bytes,true);
    io.block("trainer.actor",p->weight_alloc.mem,p->weight_alloc.total_bytes,true);
    const size_t B=(size_t)p->hypers.total_agents, H=(size_t)p->hypers.horizon;
    io.device("trainer.obs",p->env.obs.data,B*OBS_SIZE);
    io.device("trainer.actions",p->env.actions.data,B*NUM_ATNS);
    io.device("trainer.rewards",p->env.rewards.data,B); io.device("trainer.terminals",p->env.terminals.data,B);
    io.device("trainer.mask",p->env.action_mask.data,B*(size_t)p->vec->mask_size);
    io.device("trainer.philox",p->rng_states[0],B);
    io.device("trainer.beta_chain",g_cn_state[0],B*NUM_ATNS*2);
    io.device("trainer.rng_offset",p->rng_offset,2);
    io.device("trainer.pending_head_logp",g_head_roll,2*B*H*NUM_ATNS);
    io.device("trainer.pending_zone_values",g_zv_roll,2*B*H*g_sbs_zw);
    io.device("trainer.losses",p->losses,NUM_LOSSES);
    io.device("trainer.ent_coef",p->ppo_bufs.ent_coef,3);
    io.device("optimizer.lr",p->muon.lr,1);
    io.device("optimizer.role_m",g_sbs_hadam_m,(size_t)hdr.hadam_len);
    io.device("optimizer.role_v",g_sbs_hadam_v,(size_t)hdr.hadam_len);
    io.device("optimizer.role_bc",g_sbs_hadam_bc,2);
    io.device("optimizer.role_step",g_sbs_hadam_t,1);
    sbgpu::sb_full_env_io(io,hdr.ready_slot,p->hypers.horizon);
    io.host("trainer.profile",p->profile.accum,NUM_PROF);
    io.host("trainer.result",result);
    sb_full_dict(io,last);
    int logs=history->size;io.host("trainer.log_history_count",&logs);
    if(logs<0 || logs>history->capacity)sbfull::fail("log history capacity mismatch");
    if(io.reading)history->size=logs;
    for(int i=0;i<logs;++i)sb_full_dict(io,&history->items[i]);
    io.block("END",nullptr,0,false);
}
static void sb_full_save(const char* ckpt, PuffeRL* p, Ini* ini, Dict* last,
                         PufLogHistory* history, TrainResult* result) {
    if(!sb_full_supported(p) || !g_sbs_hadam_m || !p->async_boot)sbfull::fail("unsupported trainer state");
    sbfull::ck(cudaDeviceSynchronize());
    char path[4200],part[4260],mb[4200];
    if(snprintf(path,sizeof(path),"%s.full",ckpt)>=(int)sizeof(path) ||
       snprintf(mb,sizeof(mb),"%s.mb",ckpt)>=(int)sizeof(mb))sbfull::fail("path too long");
    SbFullHeader h{};memcpy(h.magic,"SBFULL2",7);h.version=2;h.header_bytes=sizeof(h);
    sb_full_sync_file(ckpt); sb_full_sync_file(mb);
    char hadam[4200];
    if(snprintf(hadam,sizeof(hadam),"%s.hadam",ckpt)>=(int)sizeof(hadam))sbfull::fail("path too long");
    sb_full_sync_file(hadam);
    sb_full_platform(h);h.config_hash=sb_full_config_hash(ini);
    h.weights_hash=sb_full_file_hash(ckpt);h.momentum_hash=sb_full_file_hash(mb);
    h.epoch=p->epoch;h.global_step=p->global_step;h.total_steps=p->hypers.total_timesteps;
    h.ready_slot=p->async_ready_slot;h.write_slot=p->async_write_slot;h.write_current=p->write_slot;h.boot=p->async_boot;
    h.hadam_len=g_sbs_hadam_len;h.hadam_off=g_sbs_hadam_off;h.last_log_step=p->last_log_step;
    // A killed container may leave a staging file and its PID may be reused.
    // Pick a free name without modifying or deleting any previous staging file.
    int fd=-1;
    for(unsigned serial=0;serial<1000000;++serial) {
        if(snprintf(part,sizeof(part),"%s.part.%ld.%u",path,(long)getpid(),serial)>=(int)sizeof(part))
            sbfull::fail("path too long");
        fd=open(part,O_CREAT|O_EXCL|O_WRONLY,0644);
        if(fd>=0)break;
        if(errno!=EEXIST)sbfull::fail("cannot create snapshot",part);
    }
    if(fd<0)sbfull::fail("no available snapshot staging name",path);
    FILE* f=fdopen(fd,"wb");if(!f)sbfull::fail("fdopen failed",part);
    if(fwrite(&h,sizeof(h),1,f)!=1)sbfull::fail("header write failed");
    sbfull::Stream io(f,false);sb_full_payload(io,p,h,last,history,result);
    const double now=wall_clock();h.uptime=now-p->start_time;h.since_log=now-p->last_log_time;
    h.checksum=0;h.checksum=sbfull::hash_add(sbfull::FNV_OFFSET,&h,sizeof(h));
    if(fseek(f,0,SEEK_SET)||fwrite(&h,sizeof(h),1,f)!=1||fflush(f)||fsync(fd)||fclose(f))
        sbfull::fail("snapshot flush failed",part);
    if(link(part,path))sbfull::fail("refusing to overwrite snapshot",path);
    if(unlink(part))sbfull::fail("cannot remove committed staging file",part);
    std::string dir(path);size_t slash=dir.find_last_of('/');dir=slash==std::string::npos?".":dir.substr(0,slash);
    int dfd=open(dir.c_str(),O_RDONLY|O_DIRECTORY);if(dfd<0||fsync(dfd)||close(dfd))sbfull::fail("directory sync failed");
    fprintf(stderr,"FULL_CHECKPOINT committed %s epoch %lld step %lld\n",path,(long long)h.epoch,(long long)h.global_step);
}
static void sb_full_load(const char* ckpt, PuffeRL* p, Ini* ini, Dict* last,
                         PufLogHistory* history, TrainResult* result) {
    if(!sb_full_supported(p) || !g_sbs_hadam_m)sbfull::fail("unsupported trainer state");
    char path[4200],mb[4200];
    if(snprintf(path,sizeof(path),"%s.full",ckpt)>=(int)sizeof(path) ||
       snprintf(mb,sizeof(mb),"%s.mb",ckpt)>=(int)sizeof(mb))sbfull::fail("path too long");
    FILE* f=fopen(path,"rb");if(!f)sbfull::fail("complete snapshot missing",path);
    SbFullHeader h{}, cur{};if(fread(&h,sizeof(h),1,f)!=1)sbfull::fail("truncated header",path);
    uint64_t hash=h.checksum;h.checksum=0;
    if(memcmp(h.magic,"SBFULL2",8)||h.version!=2||h.header_bytes!=sizeof(h)||h.executable_sha256[64]!=0||
       !memchr(h.driver_release,0,sizeof(h.driver_release))||
       hash!=sbfull::hash_add(sbfull::FNV_OFFSET,&h,sizeof(h)))sbfull::fail("invalid header",path);
    sb_full_platform(cur);
    bool changed_platform=strcmp(h.driver_release,cur.driver_release)||memcmp(h.gpu_uuid,cur.gpu_uuid,16)||
                          h.driver_version!=cur.driver_version;
    if(strcmp(h.executable_sha256,cur.executable_sha256)||(!g_portable_resume && changed_platform)||
       h.runtime_version!=cur.runtime_version||h.cublas_version!=cur.cublas_version||
       h.config_hash!=sb_full_config_hash(ini)||h.weights_hash!=sb_full_file_hash(ckpt)||h.momentum_hash!=sb_full_file_hash(mb)||
       h.total_steps!=p->hypers.total_timesteps||h.hadam_len!=g_sbs_hadam_len||h.hadam_off!=g_sbs_hadam_off||
       h.epoch<1||h.global_step>h.total_steps||h.global_step!=h.epoch*(int64_t)p->hypers.horizon*p->hypers.total_agents||
       h.ready_slot<0||h.ready_slot>1||h.write_slot!=1-h.ready_slot||h.write_current<0||h.write_current>1||h.boot!=1||
       !isfinite(h.uptime)||h.uptime<0||!isfinite(h.since_log)||h.since_log<0)
        sbfull::fail("binary, device, config or checkpoint mismatch",path);
    sbfull::preflight(f,sizeof(h));
    if (changed_platform && g_portable_resume)
        fprintf(stderr,"FULL_CHECKPOINT portable platform change: all state restored, subsequent arithmetic may differ\n");
    sbfull::ck(cudaDeviceSynchronize());
    g_full_restored=1;
    sbfull::Stream io(f,true);sb_full_payload(io,p,h,last,history,result);
    if(fgetc(f)!=EOF || fclose(f))sbfull::fail("extra snapshot bytes",path);
    p->epoch=h.epoch;p->global_step=h.global_step;p->async_ready_slot=h.ready_slot;
    p->async_write_slot=h.write_slot;p->write_slot=h.write_current;p->async_boot=h.boot;
    double now=wall_clock();p->start_time=now-h.uptime;p->last_log_time=now-h.since_log;p->last_log_step=h.last_log_step;
    sbfull::ck(cudaDeviceSynchronize());
    fprintf(stderr,"FULL_CHECKPOINT restored %s epoch %lld step %lld\n",path,(long long)h.epoch,(long long)h.global_step);
}
