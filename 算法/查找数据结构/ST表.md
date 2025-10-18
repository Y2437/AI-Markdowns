## 适用范围
区间查询,不可叠加问题(有自反性的运算),不支持修改
## 思想
预处理以每一个端点开头的以2为底数的长度的区间
将查询区间分为(可能)重叠的两个区间,求这两个区间共同得到的结果(如最大值)
## 具体实现
### 初始化
- 预处理lg2:
```C
int lg2[SIZE];

void init_lg2(){

    for (int i = 1; i <=SIZE; i++)   lg2[i]=(int)log2((double)i);

}
```
- 预处理st表
```C
void init_st(){

    for (int i = 0; i < SIZE; i++) dp[i][0]=t[i];

    for (int j = 1; j <lg2[SIZE]+1; j++)

        for (int i =0; i+(1<<(j-1))<=SIZE-1; i++)    

            dp[i][j]=MAX(dp[i][j-1],dp[i+(1<<(j-1))][j-1]);
            
}
```
$t[i]$为源数据,$dp[i][j]$意为以i为左端点,长度为$(1<<j)$的区间的最大值
显见$dp[i][0]=t[i]$
上下限很明显,注意运算优先级
dp过程先便历小长度区间,由两个小长度合并为大长度
$$ dp[i][j]=MAX(dp[i][j-1],dp[i+(1<<(j-1))][j-1]);$$
### 查询
```C
int get_max(int l,int r){

    int k=lg2[r-l+1];

    return MAX(dp[l][k],dp[r-(1<<k)+1][k]);

}
```
查询时要找到覆盖整个区间的最小的两个区间,其长度为$k=lg2[r-l+1]$
这两个区间左端点分别为$l$与$r-(1<<k)+1$
长度就为k

