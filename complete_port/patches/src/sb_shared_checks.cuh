// Checks the compiled shared policy, including derivatives and structural invariance.
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "sb_zone_reward.cuh"
#include "sb_zone_command.cuh"
extern "C" int sb_zone_credit_contract_check();

__global__ void sbs_sibling_probe_k(const float* o, float* out){
    int i=threadIdx.x;
    int node=i/(CN_ZPA*SBS_SIB);
    int rel=i%(CN_ZPA*SBS_SIB);
    out[i]=sbs_local_feature(o,0,node,17+SBS_FEAT+rel,nullptr);
}

static int sbs_policy_check_one(){
    constexpr int B=8,H=16,W=SB_MASK_WIDTH;
    cublas_init_handle();
    Arch arch=build_arch(OBS_SIZE,H,1,W,true,4);
    Allocator params={},acts={},grads={};
    Weights weights=weights_create(&arch,&params);
    Activations aa=arch_reg_train(&arch,weights,&acts,&grads,B);
    alloc_create(&params);alloc_create(&acts);alloc_create(&grads);
    assert(params.num_regs==grads.num_regs && params.total_bytes==grads.total_bytes);
    for(int i=0;i<params.num_regs;++i){
        assert(numel(params.regs[i].shape)==numel(grads.regs[i].shape));
        assert(params.regs[i].elem_size==grads.regs[i].elem_size);
        long bytes=numel(params.regs[i].shape)*params.regs[i].elem_size;
        assert(bytes%16==0 && "a tensor narrower than 16 bytes shifts every later Muon slot");
    }
    int pop_tick=0;cudaMemcpyToSymbol(d_sbs_popart_tick,&pop_tick,sizeof(pop_tick));
    ulong seed=73;g_init_logstd=-0.7f;weights_init(&arch,weights,&seed,0);
    // The production residual starts at zero. Give it nonzero output weights here so
    // finite differences exercise both the hidden matrix and the trunk gradient.
    if(g_sbs_zone_mlp){
        auto* w=(SBSWeights*)weights.decoder;
        puf_kaiming_init(&w->zone_mlp2,0.1f,seed++,0);
    }
    std::vector<float> raw(B*OBS_SIZE,0.0f);
    for(int b=0;b<B;++b){auto* o=raw.data()+b*OBS_SIZE;
        for(int s:{0,1,2,10,11}){
            o[CN_FZ*s]=1;for(int f=1;f<=6;++f)o[CN_FZ*s+f]=.1f*(s+1)-.05f*f+.01f*b;
        }
        for(int k:{0,1}){o[CN_OA+CN_FA*k]=1;for(int f=1;f<=8;++f)o[CN_OA+CN_FA*k+f]=.07f*(f+k+b);}
        o[CN_OH]=1;for(int f=1;f<=6;++f)o[CN_OH+f]=-.1f+.03f*f;
        for(int f=0;f<CN_FG;++f)o[CN_OG+f]=.05f*f;
    }
    Prec input={.shape={B,OBS_SIZE}};cudaMalloc(&input.data,raw.size()*sizeof(float));
    Float gl={.shape={B,W}},gs={.shape={B,W}},gv={.shape={B}};
    cudaMalloc(&gl.data,B*W*sizeof(float));cudaMalloc(&gs.data,B*W*sizeof(float));cudaMalloc(&gv.data,B*sizeof(float));
    // zone critic (and r98's per-zone value): the objective must reach the zone rows or their parameters
    // have no derivative to finite-difference. g_sbs_zpred is what sbs_zpred_k fills; g_sbs_zgrad is the
    // per-zone value gradient the backward consumes.
    const int zw=g_sbs_zone_critic?g_sbs_zw:0;
    std::vector<float> hz((size_t)B*(zw>0?zw:1),0.0f);
    float *zpred=nullptr,*zgrad=nullptr;
    if(g_sbs_zone_critic){
        cudaMalloc(&zpred,(size_t)B*zw*sizeof(float));
        cudaMalloc(&zgrad,(size_t)B*zw*sizeof(float));
        for(int b=0;b<B;++b)for(int s=0;s<zw;++s)hz[(size_t)b*zw+s]=.003f*((b+1)%3)+.001f*(s%5)-.002f;
        cudaMemcpy(zgrad,hz.data(),hz.size()*sizeof(float),cudaMemcpyHostToDevice);
        g_sbs_zpred=zpred;g_sbs_zgrad=zgrad;
    }
    // policy.beta: the decoder output has a second block of B rows (the v outputs), and gs is
    // their per-row gradient, so the objective weights that block with hs.
    const int blocks=g_beta?2:1;
    std::vector<float> hl(B*W),hs(B*W,0.0f),hv(B),out((size_t)blocks*B*(W+1));
    const float scale=g_sbs_zone_mlp?0.1f:0.001f;
    for(int b=0;b<B;++b){hv[b]=.05f*(b+1);for(int j=0;j<W;++j)hl[b*W+j]=scale*sinf((b+1)*(j+1));}
    if(g_beta)for(int b=0;b<B;++b)for(int j=0;j<W;++j)hs[b*W+j]=scale*cosf((b+2)*(j+1));
    cudaMemcpy(gl.data,hl.data(),hl.size()*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gv.data,hv.data(),hv.size()*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gs.data,hs.data(),hs.size()*sizeof(float),cudaMemcpyHostToDevice);
    auto forward=[&](cudaStream_t stream=0){
        Prec h=arch.encoder.forward(weights.encoder,aa.encoder,input,stream);
        return arch.decoder.forward(weights.decoder,aa.decoder,h,stream);
    };
    auto upload=[&](const std::vector<float>& src){cudaMemcpy(input.data,src.data(),src.size()*sizeof(float),cudaMemcpyHostToDevice);};
    auto read=[&](){Prec y=forward();cudaMemcpy(out.data(),y.data,out.size()*sizeof(float),cudaMemcpyDeviceToHost);};
    std::vector<float> zhost((size_t)B*(zw>0?zw:1),0.0f);
    auto objective=[&](){read();double v=0;for(int b=0;b<B;++b){v+=hv[b]*out[b*(W+1)+W];
        for(int j=0;j<W;++j)v+=hl[b*W+j]*out[b*(W+1)+j];
        if(g_beta)for(int j=0;j<W;++j)v+=hs[b*W+j]*out[(size_t)B*(W+1)+b*(W+1)+j];}
        if(g_sbs_zone_critic){
            cudaMemcpy(zhost.data(),zpred,zhost.size()*sizeof(float),cudaMemcpyDeviceToHost);
            for(size_t i=0;i<zhost.size();++i)v+=hz[i]*zhost[i];
        }
        return v;};
    upload(raw);read();
    int failures=0,checked=0;
    if(g_sbs_ahu_mlp){
        // The new branch is initially exactly zero; then give its output
        // weights nonzero values so both matrices receive finite differences.
        auto* w=(SBSWeights*)weights.decoder;
        std::vector<float> before=out;
        g_sbs_ahu_mlp=0;read();
        if(out!=before){printf("FAIL AHU branch zero-output start\n");++failures;}
        g_sbs_ahu_mlp=1;
        puf_kaiming_init(&w->ahu_mlp2,0.1f,seed++,0);
        read();int changed_ahu=0;
        for(int k=0;k<blocks;++k)for(int b=0;b<B;++b)for(int j=0;j<=W;++j){
            size_t ix=(size_t)k*B*(W+1)+(size_t)b*(W+1)+j;
            bool ahu=j>=CN_AA && j<CN_AH;
            bool present=ahu && raw[(size_t)b*OBS_SIZE+CN_OA+CN_FA*((j-CN_AA)/4)]!=0.f;
            if(!present && out[ix]!=before[ix]){printf("FAIL AHU branch changed non-AHU or absent output\n");++failures;}
            if(present && fabsf(out[ix]-before[ix])>1e-7f)++changed_ahu;
        }
        if(!changed_ahu){printf("FAIL AHU branch did not affect present actions\n");++failures;}
        printf("%s AHU branch zero start, active/absent mask and zone/plant isolation\n",failures?"FAIL":"PASS");
    }
    if(g_sbs_zone_value_residual){
        // Test the zero-output start against the same pooled critic, including
        // absent slots. Then exercise nonzero residuals in actual derivatives.
        auto* w=(SBSWeights*)weights.decoder;
        std::vector<float> z0((size_t)B*zw),z1((size_t)B*zw),before=out;
        cudaMemcpy(z1.data(),zpred,z1.size()*sizeof(float),cudaMemcpyDeviceToHost);
        g_sbs_zone_value_residual=0;read();
        cudaMemcpy(z0.data(),zpred,z0.size()*sizeof(float),cudaMemcpyDeviceToHost);
        g_sbs_zone_value_residual=1;
        if(out!=before || z0!=z1){printf("FAIL zone residual zero start\n");++failures;}
        puf_kaiming_init(&w->zvr2,0.1f,seed++,0);
        read();cudaMemcpy(z1.data(),zpred,z1.size()*sizeof(float),cudaMemcpyDeviceToHost);
        if(out!=before){printf("FAIL zone residual changed actor/building output\n");++failures;}
        int differs=0;
        for(int b=0;b<B;++b)for(int z=0;z<zw;++z){
            bool present=raw[(size_t)b*OBS_SIZE+CN_FZ*z]!=0.f;
            float d=z1[(size_t)b*zw+z]-z0[(size_t)b*zw+z];
            if(!present && d!=0.f){printf("FAIL absent zone residual\n");++failures;}
            if(present && fabsf(d)>1e-7f)++differs;
        }
        if(!differs){printf("FAIL zone residual had no present-zone effect\n");++failures;}
        printf("%s zone residual zero start, present/absent mask and actor isolation\n",failures?"FAIL":"PASS");
    }
    read();std::vector<float> baseline=out;
    Prec dh=arch.decoder.backward(weights.decoder,aa.decoder,gl,gs,gv,0);
    arch.encoder.backward(weights.encoder,aa.encoder,dh,0);cudaDeviceSynchronize();
    // Match finite differences of the real forward pass to the actual backward.
    // Cover encoder, critic, and each shared actuator matrix; skip the unused GRU
    // here because this check isolates the new encoder/decoder, not existing RNN code.
    // The last tensor is the GRU, which this forward does not use. Logstd is t==2.
    for(int t=0;t<params.num_regs-1;++t){
        if(g_sbs_zone_value_residual && t==0)continue; // local value input is intentionally detached
        if(t==2)continue; // Gaussian width does not enter this logits-only objective.
        if(t==g_sbs_popart_reg)continue; // statistics: zero backprop gradient, updated by the return rule.
        auto* wt=(float*)*params.regs[t].data_ptr;auto* gt=(float*)*grads.regs[t].data_ptr;
        long n=numel(params.regs[t].shape);
        for(int pick=0;pick<12;++pick){long index=(pick*97+3)%n;float original,analytic;
            cudaMemcpy(&original,wt+index,sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(&analytic,gt+index,sizeof(float),cudaMemcpyDeviceToHost);
            float eps=1e-3f,plus=original+eps,minus=original-eps;
            cudaMemcpy(wt+index,&plus,sizeof(float),cudaMemcpyHostToDevice);double a=objective();
            cudaMemcpy(wt+index,&minus,sizeof(float),cudaMemcpyHostToDevice);double b=objective();
            cudaMemcpy(wt+index,&original,sizeof(float),cudaMemcpyHostToDevice);
            double numeric=(a-b)/(plus-minus);++checked;
            if(!std::isfinite(analytic)||fabs(analytic-numeric)>3e-4+.015*fabs(numeric)){
                printf("FAIL shared gradient tensor %d index %ld: %.9g vs %.9g\n",t,index,(double)analytic,numeric);++failures;
            }
        }
    }
    printf("shared policy: %d finite-difference checks, %d failures\n",checked,failures);
    if(g_sbs_zone_critic){cudaFree(zpred);cudaFree(zgrad);g_sbs_zpred=nullptr;g_sbs_zgrad=nullptr;}
    // One PopArt update must leave the actor bit-identical and the value
    // unchanged to fp32 rounding. Returns -80,-80,0,0: mean -40, std 40.
    if(!g_sbs_value_shared){
        upload(raw);read();std::vector<float> before=out;
        float host_pairs[2]={-160.0f,12800.0f};
        float* pairs=nullptr;float* momentum=nullptr;
        cudaMalloc(&pairs,sizeof(host_pairs));
        cudaMalloc(&momentum,(size_t)params.total_elems*sizeof(float));
        cudaMemcpy(pairs,host_pairs,sizeof(host_pairs),cudaMemcpyHostToDevice);
        cudaMemset(momentum,0,(size_t)params.total_elems*sizeof(float));
        sbs_popart_commit(weights.decoder,(float*)params.mem,momentum,(precision_t*)params.mem,
            pairs,1,4,0);
        cudaDeviceSynchronize();
        read();
        float act=0,val=0;int width=W;
        for(int b=0;b<B;++b){
            for(int j=0;j<width;++j)act=fmaxf(act,fabsf(out[b*(width+1)+j]-before[b*(width+1)+j]));
            val=fmaxf(val,fabsf(out[b*(width+1)+width]-before[b*(width+1)+width]));
        }
        float mu=0.0f;
        cudaMemcpy(&mu,((SBSWeights*)weights.decoder)->pop.data,sizeof(float),cudaMemcpyDeviceToHost);
        int bad=!(act==0.0f&&val<1e-4f&&fabsf(mu+40.0f)<1e-3f);
        printf("%s popart preserve (actor %.9g, value %.9g, mu %.9g)\n",bad?"FAIL":"PASS",(double)act,(double)val,(double)mu);
        failures+=bad;
        cudaFree(pairs);cudaFree(momentum);
    }
    auto verify_output=[&](const char* name,const std::vector<float>& expected){
        read();float worst=0;for(size_t i=0;i<out.size();++i)worst=fmaxf(worst,fabsf(out[i]-expected[i]));
        printf("%s %s max_delta %.9g\n",worst<3e-5?"PASS":"FAIL",name,(double)worst);
        if(worst>=3e-5)++failures;
    };
    // Baseline action proposals/integrator state are deliberately excluded.
    auto changed=raw;
    for(int b=0;b<B;++b){auto* o=changed.data()+b*OBS_SIZE;
        for(int s=0;s<CN_ZS;++s){o[CN_FZ*s+7]=1000+s;o[CN_FZ*s+8]=-1000-s;}
        for(int k=0;k<CN_K;++k)for(int f=9;f<CN_FA;++f)o[CN_OA+CN_FA*k+f]=2000+f;
        for(int f=7;f<CN_FH;++f)o[CN_OH+f]=-3000-f;
    }
    upload(changed);verify_output("independent of PID-derived observation columns",baseline);
    // The sibling block is addressed by slot, so swapping two zones on one
    // handler is not an invariance. The contract is the four-number entry.
    {
        std::vector<float> obs(OBS_SIZE,0.0f);
        auto put=[&](int s,float temp,float damper,float reheat){
            obs[CN_FZ*s]=1.0f;obs[CN_FZ*s+1]=temp;obs[CN_FZ*s+5]=damper;obs[CN_FZ*s+6]=reheat;
        };
        put(0,0.0f,0.0f,0.0f);
        put(1,0.5f,-0.25f,0.5f);
        put(2,-1.0f,0.25f,-0.5f);
        obs[CN_FZ*3+1]=9.0f;obs[CN_FZ*3+5]=9.0f;obs[CN_FZ*3+6]=9.0f;
        float* dobs=nullptr;float* got=nullptr;
        constexpr int N=2*CN_ZPA*SBS_SIB;
        cudaMalloc(&dobs,obs.size()*sizeof(float));
        cudaMalloc(&got,N*sizeof(float));
        cudaMemcpy(dobs,obs.data(),obs.size()*sizeof(float),cudaMemcpyHostToDevice);
        sbs_sibling_probe_k<<<1,N>>>(dobs,got);
        std::vector<float> host(N);
        cudaMemcpy(host.data(),got,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(dobs);cudaFree(got);
        auto at=[&](int node,int j,int field){return host[(node*CN_ZPA+j)*SBS_SIB+field];};
        int bad=0;
        for(int field=0;field<SBS_SIB;++field)if(at(0,0,field)!=0.0f)++bad;
        if(at(0,1,0)!=0.5f||at(0,1,1)!=-0.25f||at(0,1,2)!=0.5f||at(0,1,3)!=1.0f)++bad;
        if(at(0,2,0)!=-1.0f||at(0,2,1)!=0.25f||at(0,2,2)!=-0.5f||at(0,2,3)!=1.0f)++bad;
        for(int j=3;j<CN_ZPA;++j)for(int field=0;field<SBS_SIB;++field)if(at(0,j,field)!=0.0f)++bad;
        if(at(1,0,0)!=0.0f||at(1,0,1)!=0.0f||at(1,0,2)!=0.0f||at(1,0,3)!=1.0f)++bad;
        for(int field=0;field<SBS_SIB;++field)if(at(1,1,field)!=0.0f)++bad;
        printf("%s sibling slot contract (temp, damper, reheat, presence)\n",bad?"FAIL":"PASS");
        failures+=bad?1:0;
    }
    changed=raw;auto expected=baseline;
    for(int b=0;b<B;++b){
        for(int f=0;f<CN_ZPA*CN_FZ;++f)std::swap(changed[b*OBS_SIZE+f],changed[b*OBS_SIZE+CN_ZPA*CN_FZ+f]);
        for(int f=0;f<CN_FA;++f)std::swap(changed[b*OBS_SIZE+CN_OA+f],changed[b*OBS_SIZE+CN_OA+CN_FA+f]);
        for(int k=0;k<blocks;++k){size_t o=(size_t)k*B*(W+1)+b*(W+1);
            for(int c=0;c<2*CN_ZPA;++c)std::swap(expected[o+c],expected[o+2*CN_ZPA+c]);
            for(int c=0;c<4;++c)std::swap(expected[o+CN_AA+c],expected[o+CN_AA+4+c]);}
    }
    upload(changed);verify_output("AHU/block permutation equivariance",expected);
    upload(raw);read();
    int padding_failures=0;
    for(int k=0;k<blocks;++k){size_t o=(size_t)k*B*(W+1);
    for(int b=0;b<B;++b)for(int s=0;s<CN_ZS;++s)if(raw[b*OBS_SIZE+CN_FZ*s]==0)
        for(int c=0;c<2;++c)if(out[o+b*(W+1)+2*s+c]!=0)++padding_failures;
    for(int b=0;b<B;++b)for(int j=CN_ACT;j<W;++j)if(out[o+b*(W+1)+j]!=0)++padding_failures;
    if(k==1)for(int b=0;b<B;++b)if(out[o+b*(W+1)+W]!=0)++padding_failures;}
    failures+=padding_failures;
    printf("%s absent slots and padded output columns are zero (failures=%d)\n",padding_failures?"FAIL":"PASS",padding_failures);
    cudaStream_t stream;cudaStreamCreate(&stream);cudaGraph_t graph;cudaGraphExec_t executable;
    cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);forward(stream);
    cudaError_t status=cudaStreamEndCapture(stream,&graph);
    if(status==cudaSuccess)status=cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0);
    if(status==cudaSuccess){cudaGraphLaunch(executable,stream);status=cudaStreamSynchronize(stream);}
    printf("%s shared forward CUDA graph: %s\n",status==cudaSuccess?"PASS":"FAIL",cudaGetErrorString(status));
    if(status!=cudaSuccess)++failures;
    printf("SHARED POLICY CHECK: %s\n",failures?"FAIL":"ALL CHECKS PASSED");
    return failures?1:0;
}

static int sbs_zone_contract(){
    int bad=sb_zone_credit_contract_check();
    const double max_pph=100, min_pph=40, ref=1, dt=300, hsp=293, csp=297;
    const double occ0=2, occ1=2, total=4;
    double p0=sb_zone_productivity(295,hsp,csp,ref,max_pph,min_pph,occ0,total,dt);
    double p1=sb_zone_productivity(290,hsp,csp,ref,max_pph,min_pph,occ1,total,dt);
    if(p0!=0.0)++bad;
    auto produced=[&](double temp,double occ){
        double dev=temp-csp; if(hsp-temp>dev)dev=hsp-temp; if(dev<0)dev=0;
        double shortfall=dev<=ref?dev/ref:1+((dev-ref)/ref)*((dev-ref)/ref);
        return max_pph*(1-shortfall)*occ*dt/3600.0;
    };
    double cum=produced(295,occ0)+produced(290,occ1);
    double max_prod=max_pph*total*dt/3600.0, min_prod=min_pph*total*dt/3600.0;
    double npr=(cum-min_prod)/(max_prod-min_prod)-1;
    if(fabs(p0+p1-npr)>1e-12)++bad;
    const double nec=0.4,ncar=0.5,pw=0.6,ew=0.2,cw=0.2,scale=1;
    double r0=sb_zone_target(p0,occ0/total,nec,ncar,pw,ew,cw,scale);
    double r1=sb_zone_target(p1,occ1/total,nec,ncar,pw,ew,cw,scale);
    double scored=scale*(pw*npr-ew*nec-cw*ncar)/(pw+ew+cw);
    if(fabs(r0+r1-scored)>1e-12)++bad;
    if(sb_zone_productivity(280,hsp,csp,ref,max_pph,min_pph,1,0,dt)!=0.0)++bad;
    if(sb_zone_target(0,0,nec,ncar,pw,ew,cw,scale)!=0.0)++bad;

    const int H=4,rows=2,obs_dim=CN_FZ*CN_ZS,n=rows*H*CN_ZS;
    std::vector<float> v(n,0.0f),rew(n,0.0f);
    v[0]=1.0f;
    rew[1*CN_ZS]=2.0f;
    v[(1*H+3)*CN_ZS]=10.0f;
    float *dv,*dr,*da,*dret; precision_t* dd;
    cudaMalloc(&dv,n*sizeof(float)); cudaMalloc(&dr,n*sizeof(float));
    cudaMalloc(&da,n*sizeof(float)); cudaMalloc(&dret,n*sizeof(float));
    cudaMalloc(&dd,(size_t)rows*H*sizeof(precision_t));
    cudaMemcpy(dv,v.data(),n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dr,rew.data(),n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(dd,0,(size_t)rows*H*sizeof(precision_t));
    sbs_zone_gae(dv,dr,dd,da,dret,rows,H,0.5f,1.0f,0);
    std::vector<float> adv(n),ret(n);
    cudaMemcpy(adv.data(),da,n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(ret.data(),dret,n*sizeof(float),cudaMemcpyDeviceToHost);
    if(fabsf(adv[0]-1.0f)>1e-5f)++bad;
    if(fabsf(ret[0]-2.0f)>1e-5f)++bad;
    if(adv[3*CN_ZS]!=0.0f)++bad;
    if(adv[1]!=0.0f)++bad;
    if(fabsf(adv[(1*H)*CN_ZS]-1.25f)>1e-4f)++bad;
    cudaFree(dv);cudaFree(dr);cudaFree(da);cudaFree(dret);cudaFree(dd);

    int saved_c=g_sbs_zone_critic,saved_s=g_sbs_value_shared,saved_e=g_sbs_value_extra;
    g_sbs_zone_critic=1; g_sbs_value_shared=1; g_sbs_value_extra=1;
    Arch arch=build_arch(OBS_SIZE,16,1,SB_MASK_WIDTH,true,4);
    Allocator params={};
    Weights weights=weights_create(&arch,&params);
    auto* sw=(SBSWeights*)weights.decoder;
    if(sw->base.weight.shape[0]!=1+CN_ZS)++bad;
    long bytes=(long)sw->base.weight.shape[0]*sw->base.weight.shape[1]*sizeof(precision_t);
    if(bytes%16!=0)++bad;
    g_sbs_zone_critic=saved_c; g_sbs_value_shared=saved_s; g_sbs_value_extra=saved_e;

    const int slots=CN_ZS;
    precision_t *rw,*robs; float *rf,*rpred,*rold,*rret,*rerr;
    cudaMalloc(&rw,SBS_ROOM*sizeof(precision_t));
    cudaMalloc(&robs,(size_t)OBS_SIZE*sizeof(precision_t));
    cudaMalloc(&rf,(size_t)slots*SBS_FEAT*sizeof(float));
    cudaMalloc(&rpred,slots*sizeof(float));
    cudaMalloc(&rold,slots*sizeof(float));
    cudaMalloc(&rret,slots*sizeof(float));
    cudaMalloc(&rerr,slots*sizeof(float));
    std::vector<float> hw(SBS_ROOM,0.f), hf((size_t)slots*SBS_FEAT,0.f), ho(OBS_SIZE,0.f);
    hw[9]=1.f; hw[SBS_FEAT]=0.5f;
    hf[9]=sinhf(1.5f);
    hf[SBS_FEAT+9]=100.f;
    ho[CN_FZ*0]=1.f;
    cudaMemcpy(rw,hw.data(),SBS_ROOM*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(rf,hf.data(),hf.size()*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(robs,ho.data(),ho.size()*sizeof(float),cudaMemcpyHostToDevice);
    sbs_room_fwd_k<<<1,slots,0>>>(robs,rf,rw,rpred,1);
    std::vector<float> got(slots);
    cudaMemcpy(got.data(),rpred,slots*sizeof(float),cudaMemcpyDeviceToHost);
    if(fabsf(got[0]-2.0f)>1e-4f)++bad;
    if(got[1]!=0.0f)++bad;
    cudaMemcpy(rold,rpred,slots*sizeof(float),cudaMemcpyDeviceToDevice);
    std::vector<float> hret(slots,0.f); hret[0]=5.f;
    cudaMemcpy(rret,hret.data(),slots*sizeof(float),cudaMemcpyHostToDevice);
    sbs_room_vloss(rpred,rold,rret,robs,rerr,1,OBS_SIZE,100.f,1.f,0);
    sbs_room_grad_k<<<1,1,0>>>(robs,rf,rerr,rw,1);
    cudaMemcpy(hw.data(),rw,SBS_ROOM*sizeof(float),cudaMemcpyDeviceToHost);
    if(fabsf(hw[9]-(-4.5f))>1e-3f)++bad;
    if(fabsf(hw[SBS_FEAT]-(-3.0f))>1e-3f)++bad;
    for(int k=0;k<SBS_FEAT;++k) if(k!=9 && hw[k]!=0.0f)++bad;
    cudaFree(rw);cudaFree(robs);cudaFree(rf);cudaFree(rpred);cudaFree(rold);cudaFree(rret);cudaFree(rerr);
    printf("%s zone critic contract (return sum, empty zero, current-step GAE, 91 value rows, room feature path)\n",bad?"FAIL":"PASS");
    return bad?1:0;
}

static int sbs_zone_command_contract(){
    int bad=0;
    auto near=[&](float got, float want){ if(!(fabsf(got-want)<=1e-5f)) ++bad; };
    float d,r,dp,run;
    /* r61_mod_band: joint proportional warm stroke; water capped by the 4 K demand reset */
    sb_zone_curve(0.0f,&d,&r); near(d,SB_ZONE_FLOOR_A); near(r,0.0f); near((d+1.0f)*0.5f,0.1f); near((r+1.0f)*0.5f,0.5f);
    sb_zone_curve(0.25f,&d,&r); near(d,SB_ZONE_FLOOR_A+0.25f*(1.0f-SB_ZONE_FLOOR_A)); near(r,0.25f); near((r+1.0f)*0.5f,0.625f);
    sb_zone_curve(0.5f,&d,&r); near(d,SB_ZONE_FLOOR_A+0.5f*(1.0f-SB_ZONE_FLOOR_A)); near(r,0.5f);
    sb_zone_curve(0.75f,&d,&r); near(d,SB_ZONE_FLOOR_A+0.75f*(1.0f-SB_ZONE_FLOOR_A)); near(r,0.75f);
    sb_zone_curve(1.0f,&d,&r); near(d,1.0f); near(r,1.0f);
    sb_zone_curve(-0.5f,&d,&r); near(d,SB_ZONE_FLOOR_A); near(r,-0.5f);
    sb_zone_curve(-1.0f,&d,&r); near(d,SB_ZONE_FLOOR_A); near(r,-1.0f);
    sb_zone_curve(2.0f,&d,&r); near(d,1.0f); near(r,1.0f);
    sb_zone_curve(-2.0f,&d,&r); near(d,SB_ZONE_FLOOR_A); near(r,-1.0f);
    sb_zone_cool(-0.5f,&d,&r); near(d,SB_ZONE_FLOOR_A+0.5f*(1.0f-SB_ZONE_FLOOR_A)); near(r,-1.0f);
    sb_zone_cool(-1.0f,&d,&r); near(d,1.0f); near(r,-1.0f);
    sb_zone_apply(-0.5f,0,&d,&r); near(d,SB_ZONE_FLOOR_A); near(r,-0.5f);
    sb_zone_apply(-0.5f,1,&d,&r); near(d,SB_ZONE_FLOOR_A+0.5f*(1.0f-SB_ZONE_FLOOR_A)); near(r,-1.0f);
    sb_zone_apply(0.0f,1,&d,&r); near(d,SB_ZONE_FLOOR_A); near(r,0.0f);
    sb_plant_from_heat(0.5f,&dp,&run); near(dp,0.0f); near(run,0.0f);
    sb_plant_from_heat(1.0f,&dp,&run); near(dp,1.0f); near(run,1.0f);
    sb_plant_from_heat(0.0f,&dp,&run); near(dp,-1.0f); near(run,-1.0f);
    sb_plant_from_heat(0.25f,&dp,&run); near(dp,-0.5f); near(run,-0.5f);
    near(sb_water_reset_cap(0.0f),-1.0f);
    near(sb_water_reset_cap(1.0f),-0.5f);
    near(sb_water_reset_cap(2.0f),0.0f);
    near(sb_water_reset_cap(4.0f),1.0f);
    near(sb_water_reset_cap(10.0f),1.0f);
    near(sb_water_reset_cap(-3.0f),-1.0f);
    near(sb_heat_fraction(0.0f),0.5f);
    near(sb_heat_fraction(-1.0f),0.0f);
    near(sb_heat_fraction(1.0f),1.0f);
    printf("%s zone command curve (joint proportional band, cool when asked, average pump, 4 K water reset)\n",bad?"FAIL":"PASS");
    return bad?1:0;
}

// C14: the IDBD update kernel against a CPU reference of the same formula, over six
// visits with a gradient schedule that is consistent, then conflicting, so the meta
// step, the [1-a g^2]+ trace decay, the eta/eta_min clamps and the decay wipe are all
// exercised. Also checks the invariants alpha in [eta_min, eta] and finite state.
static int sbs_idbd_contract(){
    int bad=0;
    const int N=4, V=6;
    precision_t *dg; float *dw,*db,*dh,*dstat,*dsum;
    cudaMalloc(&dg,N*sizeof(precision_t)); cudaMalloc(&dw,N*sizeof(float));
    cudaMalloc(&db,N*sizeof(float)); cudaMalloc(&dh,N*sizeof(float));
    cudaMalloc(&dstat,8*sizeof(float)); cudaMalloc(&dsum,sizeof(float));
    std::vector<float> w(N),b(N),h(N,0.f);
    for(int i=0;i<N;++i){w[i]=0.25f*(i+1); b[i]=logf(SBS_IDBD_ALPHA0)+0.5f*i;}
    float gs[V][N]={{1e-3f,2e-3f,-4e-4f,3e-2f},{1.2e-3f,1.6e-3f,-3e-4f,4e-2f},
        {0.9e-3f,2.2e-3f,2e-4f,2e-2f},{1.1e-3f,-2.0e-3f,5e-4f,5e-2f},
        {1.0e-3f,2.4e-3f,-1e-4f,-3e-2f},{0.8e-3f,1.8e-3f,3e-4f,1e-2f}};
    cudaMemcpy(dw,w.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dh,h.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    float stat0[8]={0,0,0,0,0,0,0,0};
    cudaMemcpy(dstat,stat0,8*sizeof(float),cudaMemcpyHostToDevice);
    for(int v=0;v<V;++v){
        cudaMemcpy(dg,gs[v],N*sizeof(float),cudaMemcpyHostToDevice);
        float ss=0.f;
        for(int i=0;i<N;++i) ss+=gs[v][i]*gs[v][i];
        cudaMemcpy(dsum,&ss,sizeof(float),cudaMemcpyHostToDevice);
        sbs_idbd_update_k<<<1,256,0>>>(dg,dw,db,dh,dsum,dstat,0.5f,N);
        // CPU reference, the exact kernel formula in the same order
        float clip=fminf(0.5f/(sqrtf(ss)+1e-6f),1.0f);
        for(int i=0;i<N;++i){
            float g=gs[v][i]*clip;
            float bv=b[i];
            bv+=SBS_IDBD_KAPPA/expf(bv)*(-g)*h[i];
            if(bv>logf(SBS_IDBD_ETA)) bv=logf(SBS_IDBD_ETA);
            if(!(bv>logf(SBS_IDBD_ETA_MIN))) bv=logf(SBS_IDBD_ETA_MIN);
            float a=expf(bv);
            float dwv=-a*g;
            w[i]+=dwv;
            h[i]=h[i]*fmaxf(0.0f,1.0f-a*g*g)+dwv;
            float eff=a*g*g;
            if(eff>SBS_IDBD_ETA){h[i]=0.f; bv+=logf(SBS_IDBD_DECAY)*g*g;}
            b[i]=bv;
        }
    }
    cudaDeviceSynchronize();
    std::vector<float> gw(N),gb(N),gh(N),gst(8,0.f);
    cudaMemcpy(gw.data(),dw,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(gb.data(),db,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(gh.data(),dh,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(gst.data(),dstat,8*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<N;++i){
        if(fabsf(gw[i]-w[i])>1e-6f)++bad;
        if(fabsf(gb[i]-b[i])>1e-4f)++bad;
        if(fabsf(gh[i]-h[i])>1e-6f)++bad;
        float a=expf(gb[i]);
        if(!(a>=SBS_IDBD_ETA_MIN*0.999f && a<=SBS_IDBD_ETA*1.001f))++bad;
        if(!isfinite(gw[i])||!isfinite(gh[i]))++bad;
    }
    if((int)gst[5]!=N*V)++bad;
    cudaFree(dg);cudaFree(dw);cudaFree(db);cudaFree(dh);cudaFree(dstat);cudaFree(dsum);
    printf("%s idbd contract (meta step, trace decay, eta/eta_min clamps, decay wipe, telemetry count)\n",bad?"FAIL":"PASS");
    return bad?1:0;
}

static int sbs_team_contract(){
    constexpr int B=2,H=16,Z=CN_ZS,D=H+SBS_LZ;
    int bad=0;g_sbs_team=1;sbs_team_alloc(B,H,B,B);
    SBSActs a={};a.role[0].input.shape[0]=B*Z;a.role[0].input.shape[1]=D;
    precision_t *obs=nullptr,*adv=nullptr;float* out=nullptr;
    assert(cudaMalloc(&a.role[0].input.data,(size_t)B*Z*D*sizeof(precision_t))==cudaSuccess);
    assert(cudaMalloc(&obs,(size_t)B*OBS_SIZE*sizeof(precision_t))==cudaSuccess);
    assert(cudaMalloc(&adv,B*sizeof(precision_t))==cudaSuccess);
    assert(cudaMalloc(&out,(size_t)B*Z*sizeof(float))==cudaSuccess);
    std::vector<float> x((size_t)B*Z*D,0.f),o((size_t)B*OBS_SIZE,0.f),w(D,0.f),res(B*Z),grad(D);
    float av[B]={2.f,-1.f};w[0]=.4f;w[1]=-.2f;
    for(int b=0;b<B;++b)for(int z:{0,1}){
        o[(size_t)b*OBS_SIZE+CN_FZ*z]=1.f;
        x[((size_t)b*Z+z)*D]=1.f+b+z;
        x[((size_t)b*Z+z)*D+1]=.5f*(1+b-z);
    }
    // Absent zones deliberately contain nonzero features: the mask must remove them.
    x[(size_t)2*D]=7.f;
    assert(cudaMemcpy(a.role[0].input.data,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(obs,o.data(),o.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(adv,av,sizeof(av),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(g_team_w.data,w.data(),w.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    sbs_team_apply(&a,adv,obs,out,B,OBS_SIZE,Z,0);
    sbs_team_fit(&a,adv,obs,B,OBS_SIZE,0);
    assert(cudaMemcpy(res.data(),out,res.size()*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(grad.data(),g_team_grad.data,grad.size()*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    double reference[D]={};
    for(int b=0;b<B;++b)for(int z=0;z<Z;++z){
        size_t i=(size_t)b*Z+z;bool present=z<2;
        double c=w[0]*x[i*D]+w[1]*x[i*D+1];
        double expected=present?av[b]-c:0.;
        if(fabs(res[i]-expected)>1e-5)++bad;
        if(present)for(int k=0;k<D;++k)reference[k]+=(c-av[b])*x[i*D+k]/4.;
    }
    for(int k=0;k<D;++k)if(fabs(grad[k]-reference[k])>1e-5)++bad;
    float lr=.001f;float* dlr=nullptr;assert(cudaMalloc(&dlr,sizeof(float))==cudaSuccess);
    assert(cudaMemcpy(dlr,&lr,sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;
    assert(cudaStreamCreate(&stream)==cudaSuccess);
    assert(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
    sbs_team_apply(&a,adv,obs,out,B,OBS_SIZE,Z,stream);
    sbs_team_fit(&a,adv,obs,B,OBS_SIZE,stream);
    sbs_team_update(dlr,stream);
    assert(cudaStreamEndCapture(stream,&graph)==cudaSuccess);
    assert(cudaGraphInstantiate(&exec,graph,0)==cudaSuccess);
    assert(cudaGraphLaunch(exec,stream)==cudaSuccess);
    assert(cudaGraphLaunch(exec,stream)==cudaSuccess);
    assert(cudaStreamSynchronize(stream)==cudaSuccess);
    std::vector<float> after(D),input_after(x.size());int steps=0;
    assert(cudaMemcpy(after.data(),g_team_w.data,D*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(input_after.data(),a.role[0].input.data,x.size()*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(&steps,g_team_step,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
    if(steps!=2 || after[0]==w[0] || input_after!=x)++bad;
    cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);cudaStreamDestroy(stream);
    cudaFree(dlr);cudaFree(a.role[0].input.data);cudaFree(obs);cudaFree(adv);cudaFree(out);
    cudaFree(g_team_w.data);cudaFree(g_team_pred.data);cudaFree(g_team_err.data);cudaFree(g_team_grad.data);
    cudaFree(g_team_m);cudaFree(g_team_v);cudaFree(g_team_bc);cudaFree(g_team_count);cudaFree(g_team_step);cudaFree(g_team_diag);
    g_team_w={};g_team_pred={};g_team_err={};g_team_grad={};g_team_m=nullptr;g_team_v=nullptr;
    g_team_bc=nullptr;g_team_count=nullptr;g_team_step=nullptr;g_team_diag=nullptr;g_sbs_team=0;
    printf("%s team baseline (masked global advantage, detached regression gradient, CUDA graph Adam)\n",bad?"FAIL":"PASS");
    return bad?1:0;
}

static int sbs_ahu_init_contract(){
    constexpr int H=16,W=SB_MASK_WIDTH;
    int oldb=g_beta,oldz=g_sbs_zone_mlp,oldc=g_sbs_zone_critic;
    int oldv=g_sbs_value_shared,olde=g_sbs_value_extra,oldh=g_sbs_ahu_mlp;
    g_beta=1;g_sbs_zone_mlp=1;g_sbs_zone_critic=1;g_sbs_value_shared=1;g_sbs_value_extra=1;
    g_sbs_ahu_mlp=0;
    Arch a0=build_arch(OBS_SIZE,H,1,W,true,4);Allocator p0={};Weights w0=weights_create(&a0,&p0);
    alloc_create(&p0);ulong seed0=75;weights_init(&a0,w0,&seed0,0);
    g_sbs_ahu_mlp=1;
    Arch a1=build_arch(OBS_SIZE,H,1,W,true,4);Allocator p1={};Weights w1=weights_create(&a1,&p1);
    alloc_create(&p1);ulong seed1=75;weights_init(&a1,w1,&seed1,0);
    auto* d=(SBSWeights*)w1.decoder;int j=0,bad=0,checked=0;
    for(int i=0;i<p0.num_regs;++i){
        while(j<p1.num_regs && (p1.regs[j].data_ptr==(void**)&d->ahu_mlp1.data || p1.regs[j].data_ptr==(void**)&d->ahu_mlp2.data))++j;
        if(j>=p1.num_regs || numel(p0.regs[i].shape)!=numel(p1.regs[j].shape) || p0.regs[i].elem_size!=p1.regs[j].elem_size){++bad;break;}
        size_t n=(size_t)numel(p0.regs[i].shape)*p0.regs[i].elem_size;
        std::vector<unsigned char> x(n),y(n);
        assert(cudaMemcpy(x.data(),*p0.regs[i].data_ptr,n,cudaMemcpyDeviceToHost)==cudaSuccess);
        assert(cudaMemcpy(y.data(),*p1.regs[j].data_ptr,n,cudaMemcpyDeviceToHost)==cudaSuccess);
        if(x!=y)++bad;++checked;++j;
    }
    if(seed0!=seed1 || p1.num_regs!=p0.num_regs+2)++bad;
    std::vector<float> out(numel(d->ahu_mlp2.shape));
    assert(cudaMemcpy(out.data(),d->ahu_mlp2.data,out.size()*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    for(float x:out)if(x!=0.f){++bad;break;}
    cudaFree(p0.mem);cudaFree(p1.mem);free(p0.regs);free(p1.regs);
    g_beta=oldb;g_sbs_zone_mlp=oldz;g_sbs_zone_critic=oldc;
    g_sbs_value_shared=oldv;g_sbs_value_extra=olde;g_sbs_ahu_mlp=oldh;
    printf("%s AHU branch independent seed, old tensors identical and output zero (%d tensors, %d failures)\n",bad?"FAIL":"PASS",checked,bad);
    return bad?1:0;
}

static int sbs_policy_check(){
    cublas_init_handle();
    int failed=sbs_zone_contract()+sbs_zone_command_contract()+sbs_idbd_contract()+sbs_team_contract()+sbs_ahu_init_contract();
    for(int shared=0;shared<=1;++shared){
        g_sbs_value_shared=shared;
        for(int extra=0;extra<=1;++extra){
            g_sbs_value_extra=extra;
            printf("CHECK value_shared=%d value_delay=%d\n",shared,extra);
            failed+=sbs_policy_check_one();
        }
    }
    // zone critic on the pooled rows (r94b) and on r98's per-zone value row: the zone rows must be
    // finite-differenced or the per-zone value parameters have no derivative in the check.
    g_sbs_value_shared=1;g_sbs_value_extra=1;g_sbs_zone_critic=1;g_sbs_zw=CN_ZS;
    for(int zv=0;zv<=1;++zv){
        g_sbs_zone_value=zv;
        printf("CHECK zone_critic=1 zone_value=%d\n",zv);
        failed+=sbs_policy_check_one();
    }
    g_sbs_zone_critic=0;g_sbs_zone_value=0;g_sbs_zw=CN_ZS;
    // policy.beta with the shared critic and the delay summary (the r76 recipe).
    g_beta=1;g_sbs_value_shared=1;g_sbs_value_extra=1;
    printf("CHECK beta value_shared=1 value_delay=1\n");
    failed+=sbs_policy_check_one();
    g_sbs_zone_critic=1;g_sbs_zone_mlp=1;
    printf("CHECK beta nonlinear zone actor and zone critic\n");
    failed+=sbs_policy_check_one();
    g_sbs_ahu_mlp=1;
    printf("CHECK beta nonlinear AHU actor, zone actor and zone critic\n");
    failed+=sbs_policy_check_one();
    g_sbs_ahu_mlp=0;
    g_sbs_zone_value_residual=1;
    printf("CHECK beta nonlinear zone actor and detached nonlinear zone value residual\n");
    failed+=sbs_policy_check_one();
    g_sbs_zone_value_residual=0;
    g_sbs_zone_mlp=0;g_sbs_zone_critic=0;
    g_beta=0;
    g_sbs_value_shared=0;g_sbs_value_extra=0;
    return failed?1:0;
}
