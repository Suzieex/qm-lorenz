"""T=5 classical baselines versus the ideal-noiseless readout workflow.

Resource units are NOT interchangeable: classical RHS evaluations and quantum
oracle calls are reported separately, never as a measured speedup ratio.
The quantum budget is deliberately optimistic: each local coordinate is read
to the FINAL tolerance, without accounting for accumulated readout errors.
"""
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import binom
from carleman import Lorenz
from long_validation import ideal_path,write_csv


def local_qae_budget(F,delta,steps,K,failure=.05):
    # Constant-success canonical QAE bound and median amplification.
    required=max(1.,4*math.pi*F/delta,2*math.pi*math.sqrt(F/delta))
    bits=math.ceil(math.log2(required));M=2**bits
    p_fail=1-8/math.pi**2
    target=failure/(3*steps)
    repeats=1
    while binom.sf(repeats//2,repeats,p_fail)>target:repeats+=2
    calls=3*steps*repeats*(2*M-1)
    return dict(phase_bits=bits,median_repeats=repeats,A_calls=calls,base_calls=K*calls)


def compare(out=Path('results/short_time')):
    out.mkdir(parents=True,exist_ok=True)
    model=Lorenz();T=5.;epsilons=[1e-3,1e-6,1e-8]
    classic=[];carleman=[];selected=[]
    for name in ['standard','attractor']:
        ref=np.load(f'results/longtime/reference_{name}.npz')
        times=ref['times'][:5001];truth=ref['state'][:5001];u0=truth[0]
        for rtol in [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13]:
            solution=solve_ivp(model.rhs,(0,T),u0,method='DOP853',rtol=rtol,
                               atol=.01*rtol,dense_output=True)
            if not solution.success:raise RuntimeError(solution.message)
            error=np.linalg.norm(solution.sol(times).T-truth,axis=1)
            classic.append(dict(initial_id=name,method='DOP853',rtol=rtol,
                accepted_steps=len(solution.t)-1,rhs_evaluations=solution.nfev,
                output_grid_points=len(times),maximum_grid_error=float(error.max()),
                final_error=float(error[-1])))
        for N in [3,5,7,9]:
            for h in [.01,.005,.002,.001]:
                path,c,D,w,alpha,tail=ideal_path(model,u0,N,h,T)
                stride=round(h/.001);error=np.linalg.norm(path-truth[::stride],axis=1)
                B_path=max(np.linalg.norm(c.lift(model.normalize(u))) for u in path)
                B_domain=math.sqrt(sum((44/model.scale)**(2*k) for k in range(N+1)))
                gamma=float(w[1:].sum());F=model.scale*B_domain*gamma
                carleman.append(dict(initial_id=name,N=N,h=h,D_N=len(c.C),
                    steps=round(T/h),K=len(w)-1,alpha=alpha,gamma=gamma,
                    B_path_max=float(B_path),B_domain=B_domain,F_domain=F,
                    maximum_grid_error=float(error.max()),final_error=float(error[-1]),
                    classical_per_step_monomials=len(c.C),
                    classical_total_monomials=round(T/h)*len(c.C)))
        for epsilon in epsilons:
            # Reserve two thirds for other errors, though quantum readout is
            # still vastly more optimistic than an accumulated trajectory bound.
            candidates=[r for r in classic if r['initial_id']==name and r['maximum_grid_error']<=epsilon/3]
            best=min(candidates,key=lambda r:r['rhs_evaluations'])
            quantum=[]
            for r in carleman:
                if r['initial_id']==name and r['maximum_grid_error']<=epsilon/3:
                    budget=local_qae_budget(r['F_domain'],epsilon,r['steps'],r['K'])
                    quantum.append((budget['base_calls'],r,budget))
            _,configuration,budget=min(quantum,key=lambda x:x[0])
            selected.append(dict(initial_id=name,target_trajectory_epsilon=epsilon,
                classical_rtol=best['rtol'],classical_rhs_evaluations=best['rhs_evaluations'],
                classical_accepted_steps=best['accepted_steps'],
                classical_measured_error=best['maximum_grid_error'],
                quantum_N=configuration['N'],quantum_h=configuration['h'],
                quantum_steps=configuration['steps'],quantum_K=configuration['K'],
                quantum_deterministic_error=configuration['maximum_grid_error'],
                quantum_local_coordinate_tolerance=epsilon,
                **{f'quantum_{key}':value for key,value in budget.items()},
                quantum_accuracy_certified=False))
    write_csv(out/'classical.csv',classic)
    write_csv(out/'carleman_grid.csv',carleman)
    write_csv(out/'comparison.csv',selected)
    (out/'assumptions.json').write_text(json.dumps(dict(T=T,epsilons=epsilons,
        model='same binary64 parameters as long-time MP reference',
        quantum_hardware_noise=False,fault_tolerance_included=False,
        quantum_budget='canonical QAE: each local coordinate tolerance equals final epsilon; accumulation ignored',
        success_probability='union bound for local estimators conditional on all inputs in radius-44 domain',
        warning='No full-trajectory quantum accuracy certificate. Oracle calls and classical RHS counts are distinct units.',
        classical_output='dense classical trajectory checked on grid .001',
        quantum_output='discrete states at relifting times; no interpolation certificate',
        work_not_counted=['quantum elementary gate synthesis and state-preparation cost',
                          'classical preprocessing and dense-output evaluation cost'],
        selection='small finite parameter grid; not a proof of globally optimal implementations'),indent=2))
    lines=['# T=5：理想无噪声量子算法与经典轨道计算','',
        '下表记录原始、未剥离高阶预测器的 workflow。新增生成规则与余项读出的单步资源见 [改进报告](IMPROVEMENTS.md)；没有用旧查询数代替新实现的评估。','',
        '当前目标分为：短时间（先取 T=5，优先不超过 10）的同初值轨道；更长时间只研究统计量。Lorenz 时间在没有额外物理标定时为无量纲时间。旧 T=30 轨道实验保留为诊断，不再作为精确轨道的优化目标。','',
        '**结论：当前振幅编码＋逐步读出重提升流程，在单条 Lorenz63 轨道的 T=5 任务上没有显示端到端量子优势；去掉硬件噪声不会去掉振幅读出精度成本。此结论不等于证明所有可能量子算法都无优势。**','',
        '## 比较方法','',
        '参考解沿用长期实验中已做任意精度细化交叉检查的轨道，截取前 5 个时间单位。经典 DOP853 直接积分三维系统，以稠密输出在间隔 .001 的网格上检查误差。扫描 rtol 并选择测得误差不超过目标的三分之一、RHS 调用数最少的设置；这是经验检查，不是区间认证。','',
        '量子侧扫描 N=3,5,7,9 和 h=.01,.005,.002,.001，要求其理想确定性误差不超过目标的三分之一，然后估算 canonical QAE 的读出预算。读出使用整个半径 44 球的有限阶 B 界；对每个局部坐标直接允许误差 epsilon，**刻意忽略多步读出误差累积**。因此下表只是对该流程非常乐观的局部精度预算，不是已经达到相同全轨道精度的量子算法。','',
        'QAE 分辨率选为大于 `max(4*pi*F/epsilon,2*pi*sqrt(F/epsilon),1)` 的最小 2 的幂；中位数重复按每次估计条件成功率至少 `8/pi^2`、所有局部估计总失败概率不超过 .05 来选择。轨迹若离开假定区域需要另处理。','',
        '## 运行结果','',
        '| 初值 | 轨道误差目标 | 经典 RHS 调用 | 经典实测最大误差 | 量子 N,h | 乐观 A/A† 调用 | 乐观基础 block 调用 |',
        '|---|---:|---:|---:|---|---:|---:|']
    for r in selected:
        lines.append(f"| {r['initial_id']} | {r['target_trajectory_epsilon']:.0e} | {r['classical_rhs_evaluations']:,} | {r['classical_measured_error']:.3e} | {r['quantum_N']}, {r['quantum_h']} | {r['quantum_A_calls']:,} | {r['quantum_base_calls']:,} |")
    lines += ['', '**两类调用不是同一计量单位，不能将其数值相除称为实际速度比。** 量子基本门、制备、控制与反射开销尚未加上；经典 RHS 每次只计算三维 Lorenz 的少量算术。经典稠密输出的额外求值成本也应计入完整实现，但它不需要高维量子振幅估计。','',
        '复现：','', '```bash','OPENBLAS_NUM_THREADS=1 python short_time_comparison.py','```','',
        '## 为什么理想量子机仍然昂贵','',
        '这里的“理想”指精确酉门、没有退相干和门噪声。Born 随机性、有限相位分辨率和反复制备查询仍存在。已知制备线路的通用振幅读出，以绝对坐标容差 delta 计，普通 shots 通常含 delta^-2，QAE 含 delta^-1；这不是可以通过理想硬件消掉的噪声项。','',
        '本流程的预算形式仍是：','', '```math',
        r'Q_A\sim m q\,\frac{F}{\delta}\log\frac{mq}{\eta},\qquad Q_C=KQ_A,','```','',
        '如果还要保证整段轨道误差，则局部 delta 一般比最终 epsilon 更小。上表没有加这项惩罚。直接假设能精确读出任意振幅，相当于添加了普通量子计算模型没有的免费实数读出 oracle。','',
        '## 与经典算法的理论比较','',
        '- Carleman 提升维数是人为引入的；原问题只有三个状态变量。经典基线应包括直接积分，而不能要求经典算法重建整个高阶提升向量。',
        '- 同一种 relifted Carleman 的三个输出可预计算为三个多项式，单步按 O(D_N) 个单项式求值；经典也可以只算三个目标行。',
        '- 对固定有限 T、解析向量场且局部解析半径有正下界，采用足够短的固定分段，高阶时间 Taylor 法可令次数随 log(1/epsilon) 增长；三维二次系统的系数卷积每段约 O(p^2) 次高精度算术。数值位数也只需随 log(1/epsilon)（加有限时间放大预算）增长。这给出此类条件下对精度的多对数经典上界，不是当前 QAE 逐步读出流程的 epsilon^-1 依赖。',
        '- 该对比只针对本任务与本流程，不是关于所有量子算法的通用不可能性定理；若输出只要求一个量子态、而不是经典坐标，任务与读出成本会改变。','',
        '## 更长时间只看统计量','',
        '建议改用有明确意义的观测量，如 z 的长期均值、能量均值、自相关函数或指定事件概率。量子优势的候选来源是大批初值／参数样本的期望估计，而不是反复读出一条三维轨道。','',
        '若有成本 C_prep 的相干统计样本制备与可计算的有界观测量，QAE 可在标准 oracle 模型下把 Monte Carlo 的约 epsilon_stat^-2 样本依赖改善为 epsilon_stat^-1。但必须计入混合时间、相关性、离散／模型偏差、样本制备及逆线路，且应与好的经典统计方法比较。它是有条件的统计优势，不是已经建立的 Lorenz63 端到端加速。','']
    Path('SHORT_TIME_COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(selected,indent=2))


if __name__=='__main__':compare()
