# 完整配点阶段：预条件、末端填充与直接逆矩阵元读出

本文是当前实验的理论审计。数值矩阵的奇异值不是区间证书；以下复杂度是所述构造的充分标度或数值资源筛选，不是量子算法的普遍下界。目标仍为 Lorenz63、给定经典初值、T=5、三个经典末端坐标，精度目标 1e-8。

## 1. 保留阶段，而不经典编译传播矩阵

以 q=g/s 表示归一化校正，r 已吸收因子 1/s。Gauss 配点给出

```math
Z_{ki}=q_k+h_k\sum_{j=1}^{s_G}a_{ij}(A_{kj}Z_{kj}+r_{kj}),\qquad
q_{k+1}=q_k+h_k\sum_jb_j(A_{kj}Z_{kj}+r_{kj}).
```

未知量为所有 Z 和 q 的线性系统 Bx=d，维数为 3M(s_G+1)。它不需要先计算 E_k 或 d_k 传播矩阵。Lorenz Jacobian 至多有八个非零项，阶段耦合的稀疏度是 O(s_G)，不随 M 增长。`increment` 形式用 Δ_ki=Z_ki-q_k 替换 Z，`quadrature` 再将 Δ_ki 乘 sqrt(b_i)；这是可实现的对角变换，不是凭数值谱范数获得的 block encoding。

经典公平基线只需每段一次 3s_G 维、一个 RHS 的阶段求解，再更新三个坐标，算术量 O(M s_G^3)，存储可限制为 O(s_G^2)，不需要稠密全局逆。量子初始系统构建虽已避免 E_k,d_k，查表编译、数据制备、求逆与读出仍需逐项计入。

## 2. 前缀预条件可利用精确消去，不能只乘归一化

令 B0 为同一阶段系统在 A=0 时的矩阵，P=B0^{-1}。它是明确的时间前缀算子：端点行对过去 RHS 求前缀和；state 形式的阶段行还加上其前面所有端点 RHS 的加权和。构造 P 不需要任何 ODE 解或局部矩阵逆。

```math
B=B_0-K,\qquad PB=I-PK.
```

如果分别编码 P 和 B，归一化为 alpha_P alpha_B；但重新实现右式，可以采用 LCU 归一化

```math
\alpha_{\rm cancelled}=1+\alpha_P\alpha_K.
```

在固定阶段数、均匀时间步下，K 中每一项都含 h，所以 alpha_K=O(h)，而 alpha_P=O(M)，故乘积可保持 O(T)，而非 O(M)。非均匀网格需要控制 M max(h_k) 及系数范围。该结论依赖实际编码的分解，并不声称所有预条件器的 alpha 都可免费降低。

还可直接对 I-PK 使用本轮已实现的 shift/diagonal LCU，得到另一明确归一化：

```math
PB=\sum_d D_d S_d,\qquad
\alpha_{\rm shift}=\sum_d\max_i|(PB)_{i,i-d}|.
```

`tradeoff_prefix.csv` 同时报告原始组合、消去后组合和直接重新编码三者。前缀后的矩阵通常变稠密，查表会增大；组合编码避免显式稠密查表，但每次调用 P,K 的门数必须保留。前缀 RHS Pd 也必须制备。因而固定 M 的 alpha 改善既不是运行时间加速证明，也不是量子求解成功。

`tradeoff_prefix_scaling_control.csv` 另用固定 Lorenz Jacobian 的线性控制例，把 M 从16增至128，检查消去后组合归一化保持常数，而未经消去的组合随 M 增长。这是真正实现分解后的正面改进，仍须与真实轨道的条件数和读出成本结合。

## 3. 局部块 Jacobi 的账不能遗漏

把每段本地阶段矩阵准确求逆可改善某些条件数。若经典编译其对前态三列及 RHS 的作用，每段需要同一 3s_G 阶段系统的四个 RHS；经典最终坐标基线只需一个 RHS。此时生成预条件系数的工作已经超过直接求解同一线性校正的工作，不能当作免费量子 oracle。

若改用相干的小块逆，必须给出它的 block encoding、归一化、误差及调用成本；不能使用诊断用的稠密逆而把这些成本记为零。本轮的 Jacobi 只是带收费的经典编译对照。

## 4. 末端填充是有效改进，但不能只看成功概率

追加 p 份末端副本，方程为 y_1=x_end、y_(i+1)=y_i。扩充系统仍线性，原解保持不变。其联合末端概率为

```math
p_{\rm end}^{(p)}=\frac{(p+1)\|x_{\rm end}\|^2}{\|x\|^2+p\|x_{\rm end}\|^2}.
```

填充能显著降低末端投影所需的放大次数。但维数、逆范数和 block encoding 也改变；不能固定原条件数只优化概率。

一个可解析控制例是标量 x'=0。L 步差分矩阵 D_L 对角为1、次对角为-1，奇异值为

```math
\sigma_j(D_L)=2\sin\frac{(2j-1)\pi}{4L+2},\quad j=1,\ldots,L.
```

原有 M 步、添加 p 步时 L=M+p，p_end=(p+1)/(M+p)，kappa=Theta(M+p)。优化 kappa/sqrt(p_end) 会给 p=Theta(M)，将未填充的 O(M^(3/2)) 改善为 O(M)。这是真实改进，但没有成为对 M 的多对数标度。它只是这类差分编码的分析，绝非 Lorenz 或所有量子算法的下界。

## 5. 直接估计逆矩阵的三个元素：不必制备完整解态

这条读出与“先归一化历史态、再后选择末端”不同。假设已经有 alpha-block encoding 的 B，且已知 beta>=||B^{-1}||_2。通过奇异值变换／Hermitian dilation，可以构造块编码

```math
V\approx\frac{B^{-1}}{2\beta}.
```

额外的2留出多项式的有界逼近余量。设能制备 |dhat>=d/||d||，对三个末端索引 e_j 分别做相干 Hadamard test，则

```math
m_j=\operatorname{Re}\langle e_j|V|\widehat d\rangle,\qquad
g_j=Cm_j,\qquad C=2s\beta\|d\|_2.
```

这确实消除了历史范数未知和末端投影概率，不需要全态层析，也不用另估计 ||x||。但它没有消除数值尺度：Hadamard 概率为 p_j=(1+m_j)/2，所以 g_j=C(2p_j-1)。物理坐标误差 epsilon_j 要求概率误差 epsilon_j/(2C)，不是 epsilon_j/C。

若 V 的算子误差<=eta，则每坐标误差<=C eta。实验取每坐标 QAE 物理预算 epsilon_r/3、eta<=epsilon_r/(12C)，从而三维总误差小于 epsilon_r。`qae_count(2C,epsilon_r/3)` 使用该概率映射，三坐标和置信度重复另计。

对一个具体通用 reciprocal-QSVT 基线，滤波次数为

```math
D=O\!\left(\alpha\beta\log\frac{\alpha\beta}{\eta}\right),\qquad
Q_{\rm AE}=\widetilde O\!\left(\frac{s\beta\|d\|_2}{\epsilon_r}\right).
```

因此上述直接矩阵元构造的查询标度为两者乘积，并乘每次 block encoding、RHS 制备和受控运算的成本。现代专用矩阵元估计可能改善该基线，不能把这个乘积当作最优复杂度或下界；尤其不能由它证明量子算法不可能加速。资源脚本中的 reciprocal_degree_proxy 省略了定理常数，仅作筛选。

本轮用 Lanczos 与稀疏 LU 估计 beta；它不是已认证的逆范数上界。因此 CSV 中 QAE 预算只是“若显示的 beta 是合法上界”的条件性数值预算，而不是新算法的严格精度证书。对输入矩阵的 alpha 则使用明确 shift LCU 的精确有理归一化，两者必须区分。

还检验了输出特定的伴随行：lambda_j=B^{-T}e_j。假设能制备其归一化态，则 g_j=s||lambda_j||||d|| Re<lambdahat_j|dhat>，其 Cauchy 读出尺度确实可小于全局 2s beta||d||。但用同一个 reciprocal block encoding 制备该行的成功幅为 ||lambda_j||/(2beta)，朴素振幅放大成本是 2beta/||lambda_j||；该成本与更小读出尺度相乘，恰好回到 2s beta||d||。本轮实际计算了三条伴随行，CSV 保留行范数、尺度和放大次数，未把这些经典伴随求解当作免费的量子制备。更强的专用伴随状态制备仍可研究，不能凭这个特定构造排除。

## 6. 离散、加载和求解误差如何合并

原非线性误差、线性离散误差、矩阵加载误差和读出误差应分别控制。若矩阵和 RHS 扰动为 ΔB、Δd，且 beta||ΔB||<1，则

```math
\|\widetilde x-x\|_2
\le\frac{\beta}{1-\beta\|\Delta B\|_2}
\left(\|\Delta d\|_2+\|\Delta B\|_2\|x\|_2\right).
```

再乘物理尺度及末端投影算子的范数；阶段／变量缩放时需使用相应逆变换。对每行 sr、每列 sc 个非零的独立逐项加载误差 eta_entry，可用 ||ΔB||<=sqrt(sr sc) eta_entry。非均匀尺度不能只检查缩放后向量误差。

此前严格非线性余项证书可以复用。另已对两份12阶段数值解重构的连续多项式作严格残差认证，覆盖整个[0,5]，见[主报告](FULLSTAGE_QLSA.md)。该证书验证的是明确的经典多项式；精确阶段线性系统、有限精度量子加载与QLSA输出还须分别与它连接，不能自动视为量子线路精度证书。查表电路的实数旋转也须另计有限门集综合误差。

## 7. 公平判据

完成下列全部项目才可称当前任务有端到端优势：新离散的目标误差得到保证、逆范数有可用上界、系数/RHS相干访问的实现及成本明确、三个坐标的范数尺度完整计入，并优于共享同一预测证书的优化经典补算。不同单位的量子查询数与经典乘法数不可相除作为速度比。

对于固定维数、固定 T 的解析 Lorenz，经典方法可提高阶数，使精度成本对 1/epsilon 为多对数（需统一局部解析半径和舍入控制）。当前通用振幅估计的 1/epsilon 读出标度无法据此获得精度指数优势；这限制的是本文的构造，不是所有可能的数字输出量子算法。

复现：`OPENBLAS_NUM_THREADS=1 python fullstage_tradeoffs.py`。数据：`results/fullstage_qlsa/tradeoff_padding.csv`、`tradeoff_prefix.csv`、`tradeoff_scalar_control.csv`。这些文件保留了变差的案例。
