较平衡的二叉搜索树
# 思想
- Treap思想:每个节点定义优先级,优先级按堆排序,这样就可以近似平衡
- FHQ思想:不按照旋转,而是按照操作更加简洁的分裂与合并来调整堆序性
- 合并思想:按照优先级(秩)合并
- 分裂思想:按照值分裂
# 具体实现
## 初始化与节点定义
```C
int root=0;
int x,y; //用于插入与删除的临时变量
typedef struct node{
    int val;
    int pri;
    int size;
    int l;
    int r;
}node;
node tr[N];
int num;
int init(int val){
    tr[++num].val=val;
    tr[num].l=0;
    tr[num].r=0;
    tr[num].pri=rand();
    tr[num].size=1;
    return num;
}
void pushup(int i){
    tr[i].size=tr[tr[i].l].size+tr[tr[i].r].size+1;
}
```
没什么好说的,注意在主函数开头加上$srand(time(NULL))$初始化种子
根初始时为1,之后可能会改变
## 分裂
```C
void split(int val,int i,int  &l,int &r){// 将编号为i的节点按照val的值分裂到l与r上
    if(i==0) l=r=0; //相当于到了NULL
    else {
		if(tr[i].val<val){  //当前值小于val要分到左边
			l=i;            //这个节点属于左边
            split(val,tr[i].r,tr[i].r,r); // 这个节点的右子节点可能会大于val,继续与r分裂
        }else{
            r=i;
            split(val,tr[i].l,l,tr[i].l);
        }
        pushup(i);
    }
}
```
总效果:最开始传递的l与r存储了分裂后的左右子树
左子树中所有元素的值小于val
右子树中所有元素的值大于val
记得最后更新pushup
## 合并
```C
int merge(int l,int r){ //合并以l与r为根的树,返回新树的根
	if(!(l&&r)) return l|r; //若有零值,返回非零的那一个
	if(tr[l].pri>tr[r].pri){ //按秩合并,大根堆,现在要以l作为根
        tr[l].r=merge(tr[l].r,r);  //合并的同时要保持val的性质,因此是右子树和r合并
        pushup(l);   //更新!!
        return l;
    }else{
        tr[r].l=merge(l,tr[r].l);
        pushup(r);
        return r;
    }
}
```
最终效果:返回l与r合并得到的树的根
这颗树是一个Treap
## 查找
一般的搜索树查找即可
```C
int find(int val,int i){
    if(!i) return i;
    if(tr[i].val>val) return find(tr[i].l);
    if(tr[i].val<val) return find(tr[i].r);
    return i;
}
```
## 插入
分裂后再合并三项即可
```C
void insert(int val){
    if(find(val)) return;
    int tmp=init(val);
    split(val,root,x,y);
    x=merge(x,tmp);
    root=merge(x,y);
}
```
注意更新根节点
## 删除
分裂为三项,抛弃中间项即可
```C
void del(int val){
    if(!find(val)) return;
    split(val,root,x,y);
    int tmp=0;
    split(val+1,y,tmp,y);
    root=merge(x,y);
}
```
## 排名与选择(互逆)
```C
int get_rank(int i,int val){//返回小于val的数的数目
    if(!i) return 0;
    if(tr[i].val>val) return get_rank(tr[i].l,val);
    if(tr[i].val<val) return get_rank(tr[i].r,val)+1+tr[tr[i].l].size;
    return tr[tr[i].l].size;
}
int select(int i,int rank){     //返回排名为rank的节点编号,没有返回0
    if(!i) return 0;
    if(tr[tr[i].l].size>rank) return select(tr[i].l,rank);
    if(tr[tr[i].l].size<rank) return select(tr[i].r,rank-tr[tr[i].l].size-1);
    if(tr[tr[i].l].size==rank) return i;
}
```
# 完整代码
```C
#include <bits/stdc++.h>

using namespace std;

#define ll long long

#define ull unsigned long long

#define ISODD(a) (((a)&1)?1:0)

#define TEST_INT(A)  printf("------>%d\n",A)

#define eps 1e-6

#define PS putchar(32)

#define NL putchar(10)

  

int read1();

void write(int);

int strread(char a[],int m);

#define N 1000005

int x,y,z;

int root=0;

typedef struct node{

    int val;

    int pri;

    int size;

    int l;

    int r;

}node;

node tr[N];

int num;

int init(int val){

    tr[++num].val=val;

    tr[num].l=0;

    tr[num].r=0;

    tr[num].pri=rand();

    tr[num].size=1;

    return num;

}

void pushup(int i){

    tr[i].size=tr[tr[i].l].size+tr[tr[i].r].size+1;

}

void split(int val,int i,int  &l,int &r){

    if(i==0) l=r=0;

    else {

        if(tr[i].val<val){

            l=i;

            split(val,tr[i].r,tr[i].r,r);

        }else{

            r=i;

            split(val,tr[i].l,l,tr[i].l);

        }

        pushup(i);

    }

}

int merge(int l,int r){

    if(!(l&&r)) return l|r;

    if(tr[l].pri>tr[r].pri){

        tr[l].r=merge(tr[l].r,r);

        pushup(l);

        return l;

    }else{

        tr[r].l=merge(l,tr[r].l);

        pushup(r);

        return r;

    }

}

int find(int val,int i){

    if(!i) return i;

    if(tr[i].val>val) return find(val,tr[i].l);

    if(tr[i].val<val) return find(val,tr[i].r);

    return i;

}

void insert(int val){

    if(find(val,root)) return;

    int tmp=init(val);

    split(val,root,x,y);

    root=merge(merge(x,tmp),y);

}

void del(int val){

    if(!find(val,root)) return;

    split(val,root,x,y);

    int tmp=0;

    split(val+1,y,tmp,y);

    root=merge(x,y);

}

int get_rank(int i,int val){//返回小于val的数的数目

    if(!i) return 0;

    if(tr[i].val>val) return get_rank(tr[i].l,val);

    if(tr[i].val<val) return get_rank(tr[i].r,val)+1+tr[tr[i].l].size;

    return tr[tr[i].l].size;

}

int select(int i,int rank){    //返回排名为rank的节点编号,没有返回0

    if(!i) return 0;

    if(tr[tr[i].l].size>rank) return select(tr[i].l,rank);

    if(tr[tr[i].l].size<rank) return select(tr[i].r,rank-tr[tr[i].l].size-1);

    return i;

}

int main() {

    srand(time(NULL));

    int q=read1();

    while (q--)

    {

        int op=read1();

        int x=read1();

        if(op==1){

            write(get_rank(root,x)+1);

        }else if(op==2){

            write(tr[select(root,x-1)].val);

        }else if(op==3){

            int r=get_rank(root,x);

            if(r==0){

                printf("-2147483647");

            }else{

                write(tr[select(root,r-1)].val);

            }

        }else if(op==4){

            int s =tr[root].size;

            int r =get_rank(root, x);

            int k =find(x,root)?(r + 1):r;

            if (k >= s) {

                printf("2147483647");

            } else {

                write(tr[select(root, k)].val);

            }

        }else{

            insert(x);

            continue;

        }

        NL;

    }

    return 0;

}

  

int strread(char a[],int m){

if(m==0){

int f=scanf(" %s",a);

return f!=-1;

}else{

int ch=getchar();

if(ch==EOF) return 0;

int l=0;

       while (ch!=EOF&&(ch==10||ch==13||!isprint(ch))) {

            ch=getchar();

        }

        while (ch!=EOF&&ch!=10&&ch!=13){

            a[l++]=ch;

            ch=getchar();

        }

a[l]=0;

return 1;

}

}

void write(int x) {

    if (x < 0) putchar('-'), x = -x;

    if (x >= 10) write(x / 10);

    putchar('0' + x % 10);

}

int read1() {

    int x = 0, w = 1;

    char ch = 0;

    while (!isdigit(ch)) {

        if (ch == '-') w = -1;

        ch = getchar();

    }

    while (isdigit(ch)) {

        x = x * 10 + (ch - '0');

        ch = getchar();

    }

    return w * x;

}
```