/* Smart-building state at a completed rollout/update boundary. Included inside
 * sbgpu after all state and kernels have been declared. Never persist device
 * pointers: the new process owns a freshly allocated GpuSim / Env layout. */
#pragma once

struct SbEnvState {
    Log log;
    float ep_ret, ep_part[SB_NPARTS_ENV], ep_worst;
    int num_agents, tag, boundary_reached, step, max_steps, start_offset, start_day;
    unsigned long long start_rng;
    unsigned int start_draws, rng;
    int seed, env_id;
};
__global__ void sb_env_state_k(Env* env, SbEnvState* state, int count, int load) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Env& e = env[i]; SbEnvState& s = state[i];
    if (load) {
        e.log=s.log; e.ep_ret=s.ep_ret; e.ep_worst=s.ep_worst;
        for (int k=0;k<SB_NPARTS_ENV;k++) e.ep_part[k]=s.ep_part[k];
        e.num_agents=s.num_agents; e.tag=s.tag; e.boundary_reached=s.boundary_reached;
        e.step=s.step; e.max_steps=s.max_steps; e.start_offset=s.start_offset; e.start_day=s.start_day;
        e.start_rng=s.start_rng; e.start_draws=s.start_draws; e.rng=s.rng; e.seed=s.seed; e.env_id=s.env_id;
    } else {
        s.log=e.log; s.ep_ret=e.ep_ret; s.ep_worst=e.ep_worst;
        for (int k=0;k<SB_NPARTS_ENV;k++) s.ep_part[k]=e.ep_part[k];
        s.num_agents=e.num_agents; s.tag=e.tag; s.boundary_reached=e.boundary_reached;
        s.step=e.step; s.max_steps=e.max_steps; s.start_offset=e.start_offset; s.start_day=e.start_day;
        s.start_rng=e.start_rng; s.start_draws=e.start_draws; s.rng=e.rng; s.seed=e.seed; s.env_id=e.env_id;
    }
}
struct SbSimState {
    int step_index, start_index, dwell;
    double pump_head, flow_factor_sum, total_reheat_w, header_res_v;
    double boiler_setpoint, boiler_return, boiler_current, boiler_step_change;
    int hws_run, pump_run, hws_heating_requests;
};
struct SbGroupMeta {
    int off, nenv, ncv, n, k, ahus, vavs, zones, actions, nw, lu_ld, rpl, ep_len;
    uint64_t blobs[SB_MAX_BLOBS];
};

static bool sb_full_env_supported() {
    return g_canon && g_zone_memory && g_zone_critic && !g_zone_adv && !g_ahu_critic &&
        !g_resid && !g_prestart && !g_plant_pair && !g_comfort_shape && !g_rew_norm && !g_rew_center &&
        !g_mg_on && !g_zcheck && !getenv("SB_TRAJ");
}

static void sb_full_env_io(sbfull::Stream& io, int ready_slot, int horizon) {
    if (!sb_full_env_supported())
        sbfull::fail("unsupported environment mode for complete snapshot");
    if (ready_slot < 0 || ready_slot > 1 || horizon <= 0 ||
        ready_slot * horizon + horizon > zm_traj_h) sbfull::fail("invalid pending rollout slot");
    int groups = (int)g_groups.size(), expected_groups = groups;
    io.host("env.group_count", &groups);
    if (groups != expected_groups) sbfull::fail("group count changed");
    std::vector<unsigned char> active;
    sb_group_save(active);
    for (int gi=0; gi<groups; ++gi) {
        SbGroup& group = g_groups[gi];
        sb_group_load(group.st);
        SbGroupMeta m{};
        m.off=group.off; m.nenv=nenv; m.ncv=ncv; m.n=n; m.k=k; m.ahus=n_ahu; m.vavs=n_vav;
        m.zones=n_zones; m.actions=n_action; m.nw=nw; m.lu_ld=lu_ld; m.rpl=lu_rpl; m.ep_len=g_ep_len;
        if (nw > SB_MAX_BLOBS) sbfull::fail("too many weathers");
        for (int w=0; w<nw; ++w) m.blobs[w]=sbfull::hash_add(sbfull::FNV_OFFSET,wblobs[w].map,wblobs[w].size);
        SbGroupMeta expected = m;
        char name[64];
        auto label = [&](const char* what) { snprintf(name,sizeof(name),"group%02d.%s",gi,what); return name; };
        io.host(label("meta"), &m);
        if (memcmp(&m,&expected,sizeof(m))) sbfull::fail("scenario, layout, or blob changed", group.name);

        std::vector<SbEnvState> es((size_t)nenv);
        SbEnvState* device_env = nullptr;
        sbfull::ck(cudaMalloc(&device_env, es.size()*sizeof(SbEnvState)));
        if (!io.reading) {
            sbfull::ck(cudaMemset(device_env,0,es.size()*sizeof(SbEnvState)));
            sb_env_state_k<<<(nenv+127)/128,128>>>(g_envs,device_env,nenv,0);
            sbfull::ck(cudaDeviceSynchronize());
            sbfull::ck(cudaMemcpy(es.data(),device_env,es.size()*sizeof(SbEnvState),cudaMemcpyDeviceToHost));
        }
        io.host(label("env"),es.data(),es.size());
        if (io.reading) {
            sbfull::ck(cudaMemcpy(device_env,es.data(),es.size()*sizeof(SbEnvState),cudaMemcpyHostToDevice));
            sb_env_state_k<<<(nenv+127)/128,128>>>(g_envs,device_env,nenv,1);
            sbfull::ck(cudaDeviceSynchronize());
            sbfull::ck(cudaMemset(device_env,0,es.size()*sizeof(SbEnvState)));
            sb_env_state_k<<<(nenv+127)/128,128>>>(g_envs,device_env,nenv,0);
            sbfull::ck(cudaDeviceSynchronize());
            std::vector<SbEnvState> verify(es.size());
            sbfull::ck(cudaMemcpy(verify.data(),device_env,es.size()*sizeof(SbEnvState),cudaMemcpyDeviceToHost));
            if (memcmp(es.data(),verify.data(),es.size()*sizeof(SbEnvState)))
                sbfull::fail("environment restore verification failed",group.name);
        }
        sbfull::ck(cudaFree(device_env));

        std::vector<GpuSim> sim((size_t)nenv);
        sbfull::ck(cudaMemcpy(sim.data(),d_sims,sim.size()*sizeof(GpuSim),cudaMemcpyDeviceToHost));
        std::vector<SbSimState> scalars((size_t)nenv);
        if (!io.reading) for (int i=0;i<nenv;++i) {
            const GpuSim& s=sim[i]; SbSimState& x=scalars[i];
            x={s.step_index,s.start_index,s.dwell,s.pump_head,s.flow_factor_sum,s.total_reheat_w,
               s.header_res_v,s.boiler_setpoint,s.boiler_return,s.boiler_current,s.boiler_step_change,
               s.hws_run,s.pump_run,s.hws_heating_requests};
        }
        io.host(label("sim_scalars"),scalars.data(),scalars.size());
        if (io.reading) {
            for (int i=0;i<nenv;++i) {
                GpuSim& s=sim[i]; const SbSimState& x=scalars[i];
                s.step_index=x.step_index; s.start_index=x.start_index; s.dwell=x.dwell; s.pump_head=x.pump_head;
                s.flow_factor_sum=x.flow_factor_sum; s.total_reheat_w=x.total_reheat_w; s.header_res_v=x.header_res_v;
                s.boiler_setpoint=x.boiler_setpoint; s.boiler_return=x.boiler_return; s.boiler_current=x.boiler_current;
                s.boiler_step_change=x.boiler_step_change; s.hws_run=x.hws_run; s.pump_run=x.pump_run;
                s.hws_heating_requests=x.hws_heating_requests;
            }
            sbfull::ck(cudaMemcpy(d_sims,sim.data(),sim.size()*sizeof(GpuSim),cudaMemcpyHostToDevice));
        }
        GpuSim& s = sim[0];
        const size_t E=nenv, A=n_ahu, V=n_vav, Z=n_zones, K=k;
#define SBI(field, ptr, count) io.device(label(field), (ptr), (count))
        SBI("temp",d_temp,(size_t)lu_ld*ncv); SBI("mass_temp",d_mass_temp,(size_t)lu_ld*n);
        SBI("rng_convection",d_rngc,E); SBI("rng_occupancy",d_rngo,E);
        SBI("occupancy_state",d_occstate,E*Z*2); SBI("occupancy",d_V,E*Z);
        SBI("calendar",d_stepv,E); SBI("native_actions",d_native,E*n_action);
        SBI("episode_count",d_epcount,E); SBI("episode_return",d_epretd,E);
        SBI("episode_display",d_epret,E); SBI("reward",d_rew,E); SBI("reward_parts",d_rparts,E*SB_NPARTS);
        SBI("obs_raw",d_obsraw,E*SB_OBS); SBI("pid",d_pid,E*2*CN_K);
        SBI("ahu_run",s.ahu_run,E*A); SBI("ahu_coolreq",s.ahu_coolreq,E*A);
        SBI("vav_mode",s.vav_mode,E*V); SBI("vav_prev",s.vav_prev_valid,E*V);
        SBI("vav_dset",s.vav_dcmd_set,E*V); SBI("vav_rset",s.vav_rcmd_set,E*V);
        SBI("ahu_recirc",s.ahu_recirc,E*A); SBI("ahu_sat",s.ahu_sat,E*A);
        SBI("ahu_fan",s.ahu_fan,E*A); SBI("ahu_flow",s.ahu_flow,E*A);
        SBI("vav_damper",s.vav_damper,E*V); SBI("vav_reheat",s.vav_reheat,E*V);
        SBI("vav_ratio",s.vav_ratio,E*V); SBI("vav_heatsp",s.vav_heatsp,E*V);
        SBI("vav_coolsp",s.vav_coolsp,E*V); SBI("vav_dcmd",s.vav_dcmd,E*V); SBI("vav_rcmd",s.vav_rcmd,E*V);
        SBI("avg_temp",s.avg_temp,E*Z); SBI("sensing",s.sensing,E*V); SBI("vav_area",s.vav_area,E*V);
        SBI("ahu_wsum",s.ahu_wsum,E*A); SBI("ahu_atotal",s.ahu_atotal,E*A); SBI("ahu_satv",s.ahu_satv,E*A);
        SBI("zone_q",s.zone_q,E*Z); SBI("zone_qdiv",s.zq_div,E*Z); SBI("sensed_mean",s.sensed_mean,E*K);
        SBI("air_g",s.air_g,E*K); SBI("air_temp",s.air_T,E*K); SBI("coil",s.coil,E*K);
        SBI("zone_sp",s.zo_sp,E*Z); SBI("zone_act",s.zo_act,E*Z);
        SBI("lu_forward_flags",d_lu_fl,(size_t)n); SBI("lu_backward_flags",d_lu_fu,(size_t)n);
        SBI("lu_tickets",d_lu_tk,2); SBI("lu_epoch",d_lu_ep,1);
        SBI("lu_work",d_lu_work,(size_t)n*lu_ld); SBI("lu_result",d_lu_z,(size_t)n*lu_ld);
#undef SBI
    }
    sb_group_load(active);
    const size_t B=(size_t)zm_total, Z=CN_ZS, T=(size_t)horizon, slot=(size_t)ready_slot;
#define SBM(field, ptr, count) io.device("zone." field,(ptr),(count))
    SBM("feat",zm_feat,B*Z*ZM_FEAT); SBM("aux",zm_aux,B*4); SBM("critic",zm_critic,B*Z*2);
    SBM("ringT",zm_ringT,B*Z*48); SBM("ringD",zm_ringD,B*Z*48); SBM("count",zm_count,B*Z);
    SBM("oat",zm_oat,B); SBM("fresh",zm_fresh,B); SBM("rng",zm_rng,B);
    SBM("pending_feat",zm_feat_tr+slot*T*B*Z*ZM_FEAT,T*B*Z*ZM_FEAT);
    SBM("pending_aux",zm_aux_tr+slot*T*B*4,T*B*4);
    SBM("pending_critic",zm_critic_tr+slot*T*B*Z*2,T*B*Z*2);
    SBM("pending_reward",zm_rz_tr+slot*T*B*zm_rzw,T*B*zm_rzw);
#undef SBM
    io.host("metrics.last_score",g_last_sc,SB_MAX_BLOBS);
#define SBSY(name, sym) do { void* p=nullptr; sbfull::ck(cudaGetSymbolAddress(&p,sym)); io.block(name,p,sizeof(sym),true); } while(0)
    SBSY("metrics.live",g_live); SBSY("metrics.live_group",g_live_grp);
    SBSY("metrics.detail",g_detail_scen); SBSY("metrics.episode",g_ep_scen);
    SBSY("metrics.start_episode",g_ep_start); SBSY("metrics.first",g_first_scen);
    SBSY("metrics.start_draws",g_start_draws); SBSY("metrics.zclip",g_znclip);
#undef SBSY
}
