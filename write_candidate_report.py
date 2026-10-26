"""Render a correctness-first search for advantage without changing the task."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/quantum_candidates')
def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def report():
    residual=read('residual_integration.csv');means=read('mean_estimation_control.csv')
    space=read('spacetime.csv');ground=json.loads((OUT/'groundstate_audit.json').read_text())
    native=json.loads((OUT/'residual_oracle.json').read_text())
    lit=json.loads((OUT/'literature_eligibility.json').read_text())
    qualified=[r for r in space if r['empirical_combined_target_screen_pass']=='True']
    needed=[r for r in residual if r['residual_correction_feasible']=='True' and r['classical_predictor_stop']=='False']
    decision=dict(task='standard Lorenz63, known classical initial data, T=5, precise classical endpoint/trajectory',
        conclusion='No same-task end-to-end quantum advantage established in this candidate search.',
        tested_candidates=['quantum residual integration','space-time QLSA/preconditioning','nonlinear effective-Hamiltonian lifting'],
        positive_query_result='QAE beats Monte Carlo in part of the conditional oracle experiment, but exact finite enumeration and symbolic integration beat that workflow here',
        residual_exact_local_certificates=36,residual_resource_cases=len(residual),
        spacetime_numerical_cases=len(space),spacetime_numerically_qualified_cases=len(qualified),
        exact_groundstate_counterexample=True,groundstate_counterexample_bounded_support=True,
        not_claimed=['No universal impossibility theorem','No full QLSA circuit or hardware timing speedup','No interval certificate for new space-time discretizations'])
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    lines=['# 新量子算法候选：检验与适用边界','',
        '**本轮保持 Lorenz63、已知初值、T=5、经典高精度轨道输出的任务，没有改成高维系统或集合统计。检验了三类新的候选：量子残差积分、全时间量子线性求解、有效 Hamiltonian 基态求解。尚未建立同任务的端到端量子优势。**','',
        '有一个正面但有限的结果：条件性均值 oracle 实验确实显示了 QAE 相对 Monte Carlo 的查询收益；换回已知多项式 Lorenz，优化后的确定性经典方法更便宜。下面分别保留正面结果、正确性失败和资源限制。','',
        '参数约定沿用全局证书：sigma=10、rho=28，beta取程序存储的binary64数8/3的精确有理值；严格代数与误差界针对这一模型。初值为(1,1,1)与(-8,8,27)，局部h取文中十进制有理数。数学上精确beta=8/3与此模型的差异若需认证，应另计参数扰动，不能隐含忽略。','',
        '## 1. 量子残差积分：有优势定理，需检查 oracle 模型','',
        '[Kacewicz 的非线性初值算法](https://arxiv.org/abs/quant-ph/0510045)将局部高阶预测后的积分交给量子均值估计。在有限光滑度 q 的函数黑盒类中，量子查询复杂度可改善经典确定性点查询的精度指数，近似从 epsilon^(-1/q) 改为 epsilon^(-1/(q+1))，另有定理中的任意小指数松弛及对数因子。该结论依赖信息访问模型；允许更强的经典线性信息时比较会改变。','',
        '本轮测试该思路的局部 Taylor/Picard 残差积分变体，不声称复现完整递归最优算法。令 P_p 为 p阶预测，使用 tau=t/h：','',
        '```math',r'R_j(\tau)=h f_j(P_p(h\tau))-\frac{d}{d\tau}P_{p,j}(h\tau),',
        r'\widetilde u(h)=P_p(h)+\int_0^1R(\tau)\,d\tau.','```','',
        '对 Lorenz 二次向量场，R 是至多2p次的**已知多项式**。因此：','',
        '```math',r'\int_0^1R_j(\tau)\,d\tau=\sum_{k=0}^{2p}\frac{r_{jk}}{k+1}.','```','',
        '经典直接积分已有系数即可得到精确有理值；或者在理想算术下用 p+1 个 Gauss 点精确积分。生成两个二次卷积只需2(p+1)^2次标量乘法。不能把它与只会 Monte Carlo 的经典算法比较后，宣布原问题具有优势。','',
        '扫描两个初值、h=.001/.005/.01、p=2/3/4/6/8/12 和三个精度目标，共36组局部数学检验、108个资源条目。Picard近似的局部全区间误差用验证盒、Bernstein范围、Jacobian界及精确指数包络控制；不是用参考解误差作为保证。这些局部证书不构成T=5量子轨道证书。','',
        '| 初值 | h | p | 局部目标 | Picard严格界近似显示 | QAE充分A调用 | 经典精确Gauss点数 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in needed:
        lines.append(f"| {r['initial']} | {r['h']} | {r['p']} | {r['epsilon']} | {float(r['Picard_bound']):.3e} | {int(r['proposed_QAE_calls']):,} | {r['gauss_nodes_exact_for_residual']} |")
    lines+=['',
        '表中只列“经典原预测未被认证可停止、且校正预算可行”的条目；其余原始行全部保留。QAE计入三个坐标、中位数重复和中点求积偏差预算。A/A†调用是逻辑oracle计量；其内部的函数计算、受控旋转、反计算及门数需另计，不能与经典算术次数直接作运行时间比。严格上界采用证书有理值及外向小数；表中科学记数仅为近似展示。','',
        '### 一个实际搭建的量子均值 oracle','',
        f"对 p=4,h=.01 的y残差搭建32点查表式 oracle：{native['qubits']}个逻辑量子比特，编译到 {native['gates']['cx']:,} 个cx、{native['gates']['u']:,} 个u门。态向量模拟的成功概率与精确有限均值差 {native['error']:.2e}。这只是明确可实现的参考oracle：其生成已花费32次经典函数求值，整张表本身就决定均值，不能声称免费加载。",'',
        '基于该成功概率，进一步精确模拟 canonical QAE 的离散输出分布与9次中位数；不是用Gaussian测量噪声替代。Monte Carlo 用同样函数值的实际方差，按相同oracle调用数比较：','',
        '| QAE相位位数 | A/A†调用 | QAE归一化RMSE | MC归一化RMSE |','|---|---:|---:|---:|']
    for r in means:lines.append(f"| {r['phase_bits']} | {int(r['QAE_A_calls']):,} | {float(r['QAE_normalized_RMSE_exact_law']):.3e} | {float(r['MC_normalized_RMSE_same_function_calls']):.3e} |")
    lines+=['',
        '**部分行有真实的条件性查询收益，但不是Lorenz端到端优势。** 确定性枚举只要32个函数值；连续积分还可以直接对已知系数求和。连续积分的RMSE另含中点偏差，不能仅看相位分辨率。该查询实验没有计算量子硬件运行时间，也没有运行全局量子ODE。','',
        '## 2. 全时间QLSA：先过滤离散误差，再看条件数','',
        '沿用先前已认证的两条预测函数 c(t)，将线性校正 g prime=A(t)g+r(t) 编成全时间线性系统。测试：隐式中点、其代数等价的前缀预条件、Volterra梯形离散、四阶Gauss，以及保留预测分段边界的十六阶Gauss。共35个数值案例。','',
        '低阶或未对齐网格的32个案例没有达到末端1e-8，不能拿它们声称同精度加速。高阶对齐对照的三个案例在数值上达到目标：','',
        '| 初值 | 时间区间数M | 全局维数 | 数值条件数 | 未填充历史的末端概率 | 数值差＋既有非线性界筛选 |','|---|---:|---:|---:|---:|---:|']
    for r in qualified:
        lines.append(f"| {r['initial']} | {r['M']} | {r['history_dimension']} | {float(r['numerical_condition_number']):.4g} | {float(r['normalized_history_endpoint_probability']):.4g} | {float(r['empirical_combined_endpoint_error_screen']):.3e} |")
    lines+=['',
        '这些是**数值筛选，不是新离散算法的区间误差证书**。本节实际运行的是经典空间时间矩阵及SVD诊断，没有构建或执行QLSA线路。参考使用同一预测函数的高精度已认证线性多项式端点，舍入及非线性余项单列。极小数值差接近参考舍入层，不能据此声称1e-15的严格精度。','',
        '高阶编译型时间系统需要每段求解24×24阶段方程、4个右端列，生成传播 E_i 和强迫 d_i。公平的经典直接法每段仅需相同阶段方程的1个右端列，按时间推进即可输出结果；不能要求经典做稠密全局逆。生成 E_i,d_i 后，经典前向传播同样是O(M)。这些预处理均在CSV中计入。','',
        '前缀预条件在测试中使条件数变差，也提高了数值范数主控量。该结果不排除更好的预条件器。范数和奇异值只是数值诊断，未被伪装成已实现的block-encoding alpha。末端概率属于未填充历史；末端填充可以改变概率和其他成本，因此它不是不可消除的下界。','',
        '另一种完整阶段QLSA可以保留各Gauss阶段未知量，避免经典生成E,d，但会扩大系统并需要重新检验条件数、oracle和读出。当前未实现这种不同方案，也没有据此宣布所有QLSA都不可能有优势。多右端重复求解可能摊销预处理，但本轮为单轨任务，且预测函数依赖初值，不能自动套用该收益。','',
        '## 3. 有效Hamiltonian：能量为零仍可能不是轨道','',
        '[Physics-informed effective Hamiltonian 方法](https://arxiv.org/abs/2504.13174)提出将方程约束编码进Hamiltonian，再作基态制备。其非线性部分需要特殊的相同因子张量积态。我们检验的是“提升后直接求低能态”的必要一致性，不是复现或反驳论文的全部Chebyshev构造。','',
        '以标准初值的隐式Euler一步 h=.01 为例，令 q=(1,x,y,z)，理想提升 Y=q q^T。三条方程线性化为：','',
        '```math',r'(1+h\sigma)Y_{01}-h\sigma Y_{02}-x_0Y_{00}=0,',
        r'(1+h)Y_{02}-h\rho Y_{01}+hY_{13}-y_0Y_{00}=0,',
        r'(1+h\beta)Y_{03}-hY_{12}-z_0Y_{00}=0.','```','',
        '对L的约束构造H=L dagger L。以下矩阵是一个精确反例：','',
        '```math',r'Y=\begin{pmatrix}1&\mu^T\\\mu&\mu\mu^T+C\end{pmatrix},\quad\mu=(1,1,1)^T,\quad',
        r'C=\begin{pmatrix}30&\beta-1&26\\\beta-1&30&0\\26&0&30\end{pmatrix}.','```','',
        f"其三条提升约束残差均为精确0，Y正定、对称、Y00=1，但rank(Y)=4；向量化后的Schmidt秩也为4，不是相同因子的张量平方。恢复的均值(1,1,1)在原隐式Euler方程中的残差范数约 {ground['spurious_mean_original_residual_norm']:.6f}。对称10维空间中H的零空间维数为7，存在非物理解。",'',
        '这甚至不能只用Lorenz有界性排除。上述矩可由以下8个点支持的概率分布产生：','',
        '```math',r'u=\mu+\sqrt{30}(s_x,s_y,s_z),\quad s_j\in\{-1,1\},',
        r'p(s)=\frac{1+\frac{\beta-1}{30}s_xs_y+\frac{26}{30}s_xs_z}{8}>0.',
        r'\max_s\|u-(0,0,38)\|^2=1461+78\sqrt{30}<44^2.','```','',
        '权重、所有二阶矩及包含关系均以有理数／整数不等式验证。**边界归一化、对称、PSD与半径44球的支持仍不足以代替rank-one约束。** 这不否定额外强制乘积一致性的算法；它指出该约束、特殊基态制备与读出必须进入正确性及复杂度分析，不能仅检查谱隙或能量。','',
        f"还检查了一个实际投影过程：从自然的初始乘积态 q0 tensor q0 出发，精确求其到ker(L)的正交投影，即该H的无限虚时演化极限。所得态仍有非零二阶子式，恢复坐标的原方程残差范数约 {ground['initial_product_projection_original_residual_norm']:.6f}。因此仅从合法乘积态出发，也不能保证无约束的基态过滤保留物理一致性。",'',
        '## 4. 新文献的适用性检查','',
        '| 方法／原文 | 对当前Lorenz任务必须补齐的条件 |','|---|---|']
    for r in lit['sources']:
        lines.append(f"| [{r['method']}]({r['source']}) | {r['Lorenz_status']} |")
    lines+=['',
        '标准Lorenz原点的正特征值为(-11+sqrt1201)/2。两个非零平衡点的Routh-Hurwitz稳定性余量也以精确有理数验证为负，所以非正实部谱的前提不成立。不能仅凭“非共振”或“无耗散条件”的标题套用结论。2026年的[一般Carleman分析](https://quantum-journal.org/papers/q-2026-10-08-2233/)扩展了适用类并修正若干先前问题，但不自动涵盖这里的混沌参数。','',
        '平移到能量中心可使线性部分稳定，却引入强迫。对本报告显示的欧氏张量表示，即使优化统一尺度／半径，既有R判据的下界仍大于1。这不排除其他度量或表示；它仅阻止把有界性直接当作已满足收敛定理。','',
        '## 5. 本轮判断及未解决的问题','',
        '- 得到的正面结果：QAE条件性查询收益被数值检验；高阶全时间离散有达到目标的数值对照；有界性与张量一致性之间的区别有精确Lorenz反例。',
        '- 未得到：标准Lorenz63单条经典高精度轨道的端到端量子优势。没有更换输出为概率分布／统计量，也没有将维数优势或黑盒优势直接转移过来。',
        '- 保留的研究空间：无需经典编译传播矩阵的完整阶段QLSA、可实现的强预条件、或真正昂贵的黑盒残差oracle。但它们必须重新通过正确性、相干访问、状态制备及经典对照检查。',
        '- 本轮不构成所有量子算法无优势的定理。特别是多初值集合、量子输入或其他观测输出属于不同任务，若研究应另立成本基线。','',
        '## 6. 可复现文件与验证','',
        '```bash','OPENBLAS_NUM_THREADS=1 python residual_quantum_candidate.py',
        'OPENBLAS_NUM_THREADS=1 python spacetime_candidate.py',
        'python lifted_groundstate_audit.py','python candidate_literature_audit.py',
        'OPENBLAS_NUM_THREADS=1 python verify_quantum_candidates.py',
        'OPENBLAS_NUM_THREADS=1 python -m unittest discover -v','python write_candidate_report.py','```','',
        '量子参考oracle需要 requirements-quantum.txt。残差界重放不调用参考ODE；空间时间矩阵与SVD检查明确仅为浮点一致性；反例为精确有理代数。谱分解和大矩阵诊断仅用于研究验证，不被宣称为算法必需成本，也不是量子硬件时间。','',
        '新增回归检查修正了旧证书重放的端点条件：请求时刻只需属于已认证区间，不必逐位等于二进制浮点网格末端（例如精确1/10与浮点.1）。该修改不改变此前误差界数值。','',
        '代码：[残差积分](residual_quantum_candidate.py)、[全时间系统](spacetime_candidate.py)、[基态一致性](lifted_groundstate_audit.py)、[文献条件](candidate_literature_audit.py)。[完整数据](results/quantum_candidates/) 保留所有失败和对照。','',
        '![候选检验](results/quantum_candidates/candidates.png)','']
    Path('NEW_QUANTUM_CANDIDATES.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(means,qualified,ground)


def plot(means,qualified,ground):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    ax=axes[0]
    for key,label in [('QAE_normalized_RMSE_exact_law','QAE, exact-law simulation'),('MC_normalized_RMSE_same_function_calls','MC, same function queries')]:
        ax.loglog([int(r['QAE_A_calls']) for r in means],[float(r[key]) for r in means],'o-',label=label)
    ax.axvline(32,color='black',ls=':',label='32 queries: exact enumeration')
    ax.set(xlabel='Conditional oracle calls',ylabel='Normalized RMSE',title='Statistical gain does not beat exact integration');ax.legend(fontsize=7)
    ax=axes[1]
    ax.bar(range(len(qualified)),[float(r['numerical_condition_number']) for r in qualified])
    ax.set_yscale('log');ax.set_xticks(range(len(qualified)),[r['initial']+'\nM='+r['M'] for r in qualified],fontsize=8)
    ax.set(ylabel='Numerical condition number',title='Accuracy-screened space-time systems')
    ax=axes[2];schmidt=ground['spurious_Schmidt_coefficients']
    ax.bar(np.arange(4)-.18,schmidt,width=.35,label='spurious zero-energy state')
    ax.bar(np.arange(4)+.18,[1,0,0,0],width=.35,label='required product-state spectrum')
    ax.set(xlabel='Schmidt index',ylabel='Schmidt coefficient',title='Zero energy does not enforce rank one');ax.legend(fontsize=7)
    for ax in axes:ax.grid(True,axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'candidates.png',dpi=160);plt.close(fig)


if __name__=='__main__':report()
