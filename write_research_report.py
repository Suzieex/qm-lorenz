"""Render the five-stage report and scientific figure from saved results."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_workflow import OUT,read


def report():
    one=json.loads((OUT/'01_summary.json').read_text())
    two=json.loads((OUT/'02_summary.json').read_text())
    circuit=json.loads((OUT/'01_representative_circuit.json').read_text())
    examples=json.loads((OUT/'01_high_precision_examples.json').read_text())
    original=read('01_selected.csv');joint=read('02_joint_selected.csv')
    classical=read('02_classical_predictors.csv')
    observations=read('03_segment_observations.csv')
    basis=read('04_weighted_circuit_resources.csv');var=read('05_variational_diagnostic.csv')
    lines=['# Lorenz63：按 1→5 执行的改进研究','',
        '本报告延续 [严格局部路径界](RESIDUAL_BOUNDS.md)，按参数优化、余项读出、分段传播、基变换、完整误差预算的顺序执行。此前 [IMPROVEMENTS.md](IMPROVEMENTS.md) 的编号对应上一轮任务；本报告采用本轮五个研究方向的编号。','',
        '**结果：误差认证与查询数有改进，但本轮没有建立单步或 T=5 的端到端量子优势。21 个可比较单步任务均有严格余项界合格的低阶经典替代。第五项已实际运行，并报告未通过的全轨道认证条件，不再标为“未激活”。**','',
        '## 保证、数值估计、资源模型的界限','',
        '- **严格数学界**：Fraction 生成元、外向区间包络、带符号路径和严格传播尾；经典预测器的 Lagrange 余项；全局传播和 Taylor 预算。参数是存储 binary64 值的精确有理数，beta 不偷换为精确 8/3。',
        '- **理想逻辑模型**：精确连续角度门、准确输入态制备和经典系数算术。QAE 的 Born 随机性、分辨率和重复次数仍计入。没有加入物理噪声或纠错；基本门以任意角度 u 和 cx 计数。',
        '- **资源模型**：canonical QAE 的充分查询数，非问题本身的查询下界；大部分扫描中的 alpha、B、Gamma 用普通浮点显示与估算。第五项额外以有理不等式重放检查所列 QAE 充分条件。',
        '- **数值观测**：120 位 ODE/Carleman 比较、binary64 分段实验、数值变分方程均单列。即使观测精度很高，也不代替数学保证。',
        '- **成本单位**：量子 oracle 调用、基本门数、经典卷积乘法和有理证书预处理分别报告，不把不同单位直接相除称为速度比。','',
        '表格中的科学记数为近似显示；严格保证使用证书中的精确有理值和外向十进制值。高精度参考的细化差为0只表示当前数值分辨率下未发现差异，不是数学误差为0。','',
        '## 1. 用新界重新优化中心、尺度、N、h','',
        f"扫描两个初值、中心 energy/rho/initial、S=16/32/64、N=1,...,6、h=.001/.005/.01/.02 和目标 1e-4/1e-6/1e-8。实际计算 {one['configurations']} 行，其中 {one['feasible']} 行通过截断预算，为相同初值、h、目标选出 {one['selected']} 个候选。吸引子初值 h=.02 的局部包络未验证成功，相关配置未被接受；不是证明真实解不存在。",'',
        '取 u=c+diag(s)v，归一化场从原方程精确代数构造。每个规则 f_j(v)=a_(j,p)v^p 对应生成元权 a_(j,p)*n_j。原生阈值计数编译器对该规则使用容量 N-max(|p|-1,0)，同号对角项合并。其实际 LCU 归一化为：','',
        '```math',
        r'\alpha=N\max_j|J_{jj}|+\sum_{(j,p)\ \mathrm{offdiag}}[N-\max(|p|-1,0)]_+|a_{j,p}|.',
        '```','',
        '这来自生成规则的具体实现，不是用数值谱范数替换 alpha。输入提升范数 B 使用精确平方和及向上平方根。选 K 使物理 Taylor 尾不超过 epsilon/6，接受截断界<=epsilon/3，读出分配 epsilon/3。剩余预算在当前理想模型中保留，不能自动视作已认证的有限精度实现误差。','',
        '以相同步长选取最低的基础 block 调用数；跨步长仅作 calls/h 筛选，不冒充整条轨道认证。','',
        '| 初值，h=.001，目标1e-8 | 中心 | S | N | K | 理想截断严格界 | 基础 block 调用模型 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in original:
        if float(r['h'])==.001 and float(r['epsilon'])==1e-8:
            lines.append(f"| {r['initial']} | {r['center']} | {r['S0']} | {r['N']} | {r['K']} | {float(r['truncation_upper']):.3e} | {int(r['base_calls']):,} |")
    lines+=['', '局部中心 c=u0 很有效：初始归一化变量为零，顶层残差还含小的局部位移因子。但每次改变中心都要重新更新系数与制备参数，不能把这一步隐藏为免费 oracle。','',
        '120 位独立数值检查：','',
        '| 案例 | N | h | 严格截断界 | 理想截断实测 | 界/实测 |','|---|---:|---:|---:|---:|---:|']
    for r in examples:
        lines.append(f"| {r['case']} | {r['N']} | {r['h']} | {float(r['certified_truncation']):.4e} | {float(r['observed_ideal_error']):.4e} | {float(r['ratio']):.3f} |")
    lines+=['',
        f"另实际编译了 standard、h=.001、N={circuit['N']}、K={circuit['K']} 的完整 Hadamard A：{circuit['qubits']} 个逻辑量子比特，{circuit['gates']['cx']:,} 个 cx 和 {circuit['gates']['u']:,} 个 u 门。包含提升态制备和受控稀疏 LCU，不含外层 QAE 的额外控制、反射及其综合。它是当前 Qiskit 通用受控分解的资源，不是最优门数或硬件时间下界。",'',
        '各入选配置还记录基础编码门数。基础门数×调用数只标为资源乘积；不能当成完整受控 QAE 门数。为避免递归展开的巨大重复序列化，不保存大型 QPY；完整构建入口和资源清单可重现该电路。','',
        '## 2. 低阶预测后读高阶余项，并公平允许经典停止','',
        '对 p 阶预测，原生余项 LCU 编码：','',
        '```math',r'R_{p,K}(h)=\sum_{k=p+1}^{K}\frac{h^k C_N^k}{k!},\qquad',
        r'\Gamma_p=\sum_{k=p+1}^{K}\frac{(\alpha h)^k}{k!},\qquad',
        r'F_j=\frac{s_j}{w_1}B_w\Gamma_p.', '```','',
        '这里不只宣称余项小，而是从有实际 PREP/SELECT 的 LCU 得到 Gamma_p。若每个物理坐标容差为 delta_j，canonical QAE 取 M>=max(4*pi*F_j/delta_j,2*pi*sqrt(F_j/delta_j),1) 的 2 的幂；每次估计用 2M-1 次 A/A†，再中位数放大。三个坐标都计入。', '',
        '**阶数边界：一般二次场中，第 p 个时间导数可达 p+1 次，故只在 p<=N-1 时保证 Carleman 低阶行等于原 ODE 的 Taylor 预测器。不能直接假设 p=N 也相同。** 边界反例已写入测试。','',
        '经典预测器采用精确递推：','',
        '```math',r'P_p(h)=u_0+\sum_{k=1}^{p}\frac{h^k}{k!}L_f^k u(u_0),\qquad',
        r'|u_j(h)-P_{p,j}(h)|\le\frac{h^{p+1}}{(p+1)!}\sup_{u\in X}|L_f^{p+1}u_j(u)|.', '```','',
        'X 是验证包络；Lie 多项式在 u-u0 中展开并以有理区间求值，避免直接在大物理坐标上取粗界。预测器满足完整 epsilon 时，经典方法停止；量子侧不能继续读余项再将其算作必要收益。经典在线系数卷积乘法为 p(p+1)，证书的符号预处理乘法数、项数和系数位数另外记录。精确有理保证不等于 binary64 计算自动满足该保证。','',
        f"第 1 项入选点的显式目标行后端比较有 {two['candidates']} 个条目；进一步对全部确定性可行配置重新联合优化 p，得到 {two['joint_candidates']} 个余项候选和 {two['joint_nonstop_selected']} 个未触发其自身经典停止的量子候选。这样没有把 p=0 的优胜参数冻结后宣称余项最优。",'',
        '| 初值，h=.001，目标1e-8 | 余项候选 N,p | F | 基础 block 调用模型 | 经典可停止阶数 | 经典卷积乘法 |','|---|---|---:|---:|---:|---:|']
    for r in joint:
        if float(r['h'])==.001 and float(r['epsilon'])==1e-8:
            c=next(c for c in classical if c['initial']==r['initial'] and float(c['h'])==.001 and float(c['epsilon'])==1e-8)
            lines.append(f"| {r['initial']} | {r['N']},{r['p']} | {float(r['F']):.3e} | {int(r['base_calls']):,} | {c['certified_order']} | {c['convolution_products']} |")
    lines+=['',f"**全部 21 个入选任务都有可认证的经典替代，最高阶为 {two['max_classical_order']}。** 上表只列在线卷积，完整预处理计数见 CSV；不据表中不同单位作速度比。",'',
        '目标行后端也保留：先计算 r_j=e_j^T R，再显式制备 r_j/||r_j||，读出因子为 (s_j/w1)*B_w*||r_j||。其预计算上界按三个目标行 3K*nnz(C) 个稀疏乘加记录，并列出态制备网格大小；不能假装它与原生稀疏 LCU 有相同预处理成本。若 c=u0 使 phi 是常数基态，预计算目标行的常数列就已经给出了答案，量子读出在这个后端没有必要。','',
        '## 3. 延长单段，减少读出与重提升','',
        '对 N=2,...,6 和段长 H=.001/.005/.01/.02/.05/.1 重新认证单段误差，并比较一次连续提升与每 .001 重提升。单段证书不要求构造每个中间经典输出，但 H 越长，截断误差和 Gamma 通常越大。验证失败行保留为失败，不用参考轨道最大值补成保证。','',
        '| 初值 | 中心 | N | 总段长 | 连续提升实测误差 | 每.001重提升实测误差 |','|---|---|---:|---:|---:|---:|']
    for r in observations:
        if r['N']=='3':lines.append(f"| {r['initial']} | {r['center']} | 3 | {r['H']} | {float(r['continuous_lift_observed_error']):.3e} | {float(r['relifted_observed_error']):.3e} |")
    lines+=['',
        '这里是 binary64 数值观测；其对应单段严格界在 `03_segment_scan.csv`。文件中的 T5 查询数只是乐观筛选：每段直接使用最终 epsilon，未加传播惩罚；真正全流程预算见第 5 项。','',
        'local 行使用前两项的改进：连续段以初值为固定中心，重提升对照在每个子步更新局部中心，均取S=16。局部连续段的严格界另写在观测文件的 continuous_truncation_upper 列，不与数值观测混为一谈。`03_optimized_segments.csv` 和 `03_best_stationary_proxy.csv` 进一步继承第2项的 N,p,中心,尺度候选；其中把第一段成本乘以段数只是静态筛选，并未假设沿轨道的 B、Gamma 或可行性保持不变。','',
        '一个正面结果：standard 的局部中心 N=3、H=.01 的严格截断界约 2.14e-12，允许在这个初始段内把 .001 的十个内部步合成一次读出。但 Gamma 和查询数也会增长，因此这证明了减少读出次数的精度可行性，没有证明净成本或整条轨道优势。','',
        '减少内部测量有代价，不能把连续截断演化等同于重提升。也没有假设可用单份未知振幅态免费实现非线性重提升。对于更长段，首先应联合提高 N、优化中心和读出因子，再谈减少测量后的净收益。','',
        '## 4. 加权基、各向异性尺度、Chebyshev 基','',
        '单项式层权 w_k>0 给出 phi_w=W phi、C_w=W C W^-1；遗漏耦合必须同时改为 W_N G W_(N+1)^-1，物理恢复因子为 s_j/w1。固定中心下，尺度和层权是对角相似变换，理想截断物理解不变；有限路径界主部也相应抵消，但传播尾和实际编码成本会改变。','',
        '本轮实际实现的加权编码为两个度数对角滤波器包围原生编码：','',
        '```math',r'D_L=W/w_{\max},\quad D_R=w_{\min}W^{-1},\qquad',
        r'D_L(C/\alpha_0)D_R=C_w/[\alpha_0(w_{\max}/w_{\min})].','```','',
        '每个滤波器计算总次数、用已知角度受控旋转、再反计算总次数，并保留独立失败标志。因此 alpha_w=alpha0*kappa(W) 是该实现的真实归一化。它是保守的具体实现，不排除未来逐规则的加权编码更好。','',
        '扫描 N=2,3,4、四种尺度组合、层权 $1$、$1/k!$、$1/(k+1)$、$k!$ 和两个初值。以下是 N=2 的实际编译资源：','',
        '| 编码 | alpha | 逻辑量子比特 | cx | u |','|---|---:|---:|---:|---:|']
    for r in basis:lines.append(f"| {r['basis']} | {r['alpha'] or '不适用：目标行后端'} | {r['qubits']} | {int(r['cx']):,} | {int(r['u']):,} |")
    lines+=['',
        '**Chebyshev 不是仅换一个范数。** 代码从 L_f(T_a) 重新展开，独立构造保留矩阵和 N+1 层残差，再作同样的带符号传播和区间包络。因此没有把单项式证书直接套用。已实现通用系数态 Hadamard 读出，并计入目标行的经典计算；没有声称已得到 Chebyshev 的廉价稀疏算术 alpha。','',
        '其目标行系数级数使用 sqrt(||C||1*||C||infinity) 作**经典级数余项主控量**，不是把它当作免费的 block-encoding 归一化。这一区别在 CSV 列名中明确标识。','',
        '本轮 Chebyshev 截断界明显较差。例如 standard,N=2,h=.005 的界约 .974。原因之一是高阶 T_k(v) 在局部小 v 区域不必小，舍弃高层不能再靠小单项式因子控制。正交性本身不足以改善这一截断方案；这不是对所有正交基或闭合方案的否定。','',
        '基变换 CSV 的 deterministic_feasible 列检查截断预算；未通过的行即使列出了读出成本，也不表示已得到符合精度的算法。','',
        '## 5. 初值、每段精度、重提升与全流程稳定性','',
        '本项已经执行。提出的完整理想算法在每次读出后投影到以 (0,0,38) 为中心的半径44凸球，防止读出误差破坏统一域条件。投影相对球内真实点非扩张。对该域，Lorenz Jacobian 的对称部给出单侧 Lipschitz 常数 L<=21；它不是最大 Lyapunov 指数。','',
        '若段 i 的局部截断、Taylor、读出界分别是 d_i,t_i,delta_i，初值误差为 e0，则：','',
        '```math',r'e_m\le e^{LT}e_0+\sum_{i=0}^{m-1}e^{L(T-t_{i+1})}(d_i+t_i+\delta_i).','```','',
        '统一情况下取 W=m*exp(LT)，为初值、截断、Taylor、读出各分配 epsilon/4：e0<=epsilon/(4exp(LT))，d,t,delta<=epsilon/(4W)。本轮以精确有理级数包络 exp(105)，并认证全局残差和 Taylor 预算；不是用采样轨道代替统一域。','',
        '对 N=3/5/7/9、h=.001/.005/.01/.02、三个目标，**48 个全流程候选都未通过统一截断预算**。因此列出的巨大 QAE 数只是“若要满足该充分读出预算”的资源，不代表整体算法已经达到目标。该失败不能推出真实误差很大。','',
        '完整成本要同时报告量子门与经典算术，而不是只报告一次查询。令 q_ij=r_ij(2M_ij-1)，则本实现的成本结构为：','',
        '```math',r'Q_C=\sum_{i,j}K_iq_{ij},\qquad',
        r'G_{\rm total}=\sum_{i,j}q_{ij}\,[G_{A,i}^{\rm controlled}+G_{{\rm reflection},i}],\qquad',
        r'G_{A,i}=O(K_iG_{C,i}^{\rm controlled}+G_{\Gamma,i}+G_{\phi,i}),',
        r'C_{\rm classical}=\sum_i O(p_i^2+N_iD_{N_i}+C_{{\rm certificate},i}+C_{{\rm compile},i})',
        r'\quad+\sum_{i\ \mathrm{row\ backend}}O(3K_i\operatorname{nnz}(C_i)).','```','',
        '原生指数网格的 b=ceil(log2(N+2))，当前通用输入制备为 O(2^(3b)*poly(b)) 门，不能假设免费 QRAM；常数基态输入是可单独简化的特例。原生 G_C 来自算术稀疏线路，目标行后端则以两次系数态制备代替 K 次基础编码。经典预处理、证书和编译只有在相同系数、中心、步长、阶数反复复用时才能摊销。表中经典卷积乘法是标量运算计数，有理数位成本并未假设为硬件常数。外层相位 QAE 另需相位寄存器和逆 QFT，若显式统计完整基本门还应加这些门；表中查询模型没有把它们当作已包含。','',
        '另外，优化连续精度分配：令 w_i 为传播权，c_i 为估计成本系数，忽略整数取整及中位数重复，有','',
        '```math',r'\min_{\delta_i>0}\sum_i\frac{c_i}{\delta_i}\quad\mathrm{s.t.}\quad\sum_iw_i\delta_i\le\epsilon_r,',
        r'\delta_i=\frac{\epsilon_r\sqrt{c_i/w_i}}{\sum_j\sqrt{c_jw_j}},\qquad',
        r'Q_{\min}=\frac{(\sum_i\sqrt{c_iw_i})^2}{\epsilon_r}.','```','',
        '在 c_i 相同、w_i=exp(21(T-t_i)) 的数值模型中，相对均匀精度可改善约 26 倍，但远不足以消除 exp(105) 的保守惩罚。其比较是相同量子资源模型之间的比较，不是量子/经典速度比。','',
        '为了判断稳定性界还有多大改进空间，另积分了终端变分方程 B(t)=D Phi_(5,t)，B prime=-B J(u(t))，B(5)=I：','',
        '| 初值，h=.001 | 数值初值终端放大 | 数值终端权重和 | 全局统一权重上界量级 |','|---|---:|---:|---:|']
    for r in var:
        if float(r['h'])==.001:lines.append(f"| {r['initial']} | {float(r['terminal_initial_amplification']):.4g} | {float(r['terminal_weight_sum']):.4g} | 2.0e49 |")
    lines+=['',
        '**这些变分结果只描述参考轨道上的数值线性化终端敏感性。** 没有包络非线性偏离和整个时间区间，不能直接替换 exp(21T) 作全轨道认证。它们指出下一项真正有价值的工作是：验证局部变分传播子，并控制邻域中的 Jacobian 变化，或采用随轨道变化的度量。仅用数值最大 Lyapunov 指数替换 L 不构成证明。','',
        '## 当前结论与适用边界','',
        '- 有实际收益：局部平移显著收紧误差；余项 LCU 降低了同一量子流程的查询模型；所有基变换成本保持显式；第5项已运行并定位认证瓶颈。',
        '- 没有得到：单步量子优势、T=5 全轨道量子优势、有限门精度的完整实现认证，或所有可能 Lorenz 量子算法都无优势的定理。',
        '- 在未知振幅的标准 oracle 模型中，固定读出因子下的高精度读出最坏情形需要与 1/epsilon 同阶的查询；但 Lorenz 系数已知、可经典计算，不能把该 oracle 下界直接宣布为 Lorenz 问题的普遍下界。',
        '- 对固定短时间解析 ODE，经典高阶 Taylor 的阶数可随 log(1/epsilon) 增长。若增加预测阶数已经让经典法满足目标，量子补算不能作为加速证据。','',
        '## 复现与文件','',
        '```bash','OPENBLAS_NUM_THREADS=1 python research_workflow.py --stage 1',
        'OPENBLAS_NUM_THREADS=1 python research_workflow.py --stage 2',
        'OPENBLAS_NUM_THREADS=1 python research_workflow.py --stage 3',
        'OPENBLAS_NUM_THREADS=1 python research_workflow.py --stage 4',
        'OPENBLAS_NUM_THREADS=1 python research_workflow.py --stage 5',
        'OPENBLAS_NUM_THREADS=1 python research_workflow.py --resources-only',
        'OPENBLAS_NUM_THREADS=1 python research_workflow.py --examples-only',
        'OPENBLAS_NUM_THREADS=1 python verify_research_certificates.py',
        'OPENBLAS_NUM_THREADS=1 python -m unittest discover -v',
        'python write_research_report.py','```','',
        '`--resources-only` 会对完整受控线路作实际通用门分解，内存可达数 GB，耗时显著高于其余阶段；这是经典编译开销，也需要计入部署。没有以模拟器运行时间代表量子硬件速度。','',
        '严格界实现见 [research_bounds.py](research_bounds.py)，原生加权及通用目标行线路见 [weighted_compiler.py](weighted_compiler.py)，顺序实验见 [research_workflow.py](research_workflow.py)。全部 CSV、证书和分类元数据见 [results/research_sequence](results/research_sequence/)。','',
        '![研究结果](results/research_sequence/research_summary.png)','']
    Path('RESEARCH_SEQUENCE.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(original,joint,observations,basis,var)


def plot(original,joint,observations,basis,var):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    ax=axes[0,0];keys=[('standard',.001),('attractor',.001),('standard',.01),('attractor',.01)]
    for offset,rows,label in [(-.18,original,'increment'),(.18,joint,'optimized remainder')]:
        values=[int(next(r for r in rows if r['initial']==n and float(r['h'])==h and float(r['epsilon'])==1e-8)['base_calls']) for n,h in keys]
        ax.bar(np.arange(4)+offset,values,width=.35,label=label)
    ax.set_yscale('log');ax.set_xticks(range(4),[f'{n}\nh={h}' for n,h in keys],fontsize=8)
    ax.set_ylabel('Base block calls (resource model)');ax.set_title('Single-step quantum-to-quantum comparison');ax.legend(fontsize=8)
    ax=axes[0,1];rows=[r for r in observations if r['initial']=='standard' and r['N']=='3' and r['center']=='energy']
    for key,label in [('continuous_lift_observed_error','continuous lift'),('relifted_observed_error','relift every .001')]:
        ax.loglog([float(r['H']) for r in rows],[float(r[key]) for r in rows],'o-',label=label)
    ax.set(xlabel='Segment length',ylabel='Observed numerical vector error',title='N=3: fewer relifts changes the error');ax.legend(fontsize=8)
    ax=axes[1,0];rows=[r for r in basis if r['alpha']]
    ax.bar(range(len(rows)),[float(r['alpha']) for r in rows]);ax.set_xticks(range(len(rows)),[r['basis'] for r in rows],rotation=15,fontsize=8)
    ax.set(ylabel='Implemented alpha',title='Layer weighting pays diagonal-filter normalization')
    ax=axes[1,1];rows=[r for r in var if float(r['h'])==.001]
    ax.bar([0,1],[float(r['terminal_weight_sum']) for r in rows],label='numerical terminal linearization')
    ax.bar([2],[5000*math.exp(105)],label='certified global weight upper bound')
    ax.set_yscale('log');ax.set_xticks([0,1,2],['standard','attractor','global domain'])
    ax.set(ylabel='Propagation weight sum',title='Numerical sensitivity is not a nonlinear certificate');ax.legend(fontsize=7)
    for ax in axes.flat:ax.grid(True,axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'research_summary.png',dpi=160);plt.close(fig)


if __name__=='__main__':report()
