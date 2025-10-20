

# Lab 02 Assignment

> 班级：242111
> 
> 学号：24373169
> 
> 姓名：杨小乐

## Question 1

![image-20251018114307968](E:\a笔记\Markdown\Java\实验报告\Lab2\Lab2问答报告.assets\image-20251018114307968.png)

1.第4行会导致错误,因为没有初始化m就将其使用了

2.回答如下:

- x不需要,因为是实例变量 ,会自动进行默认初始化

- m需要,是局部变量,不会自动进行默认初始化,在局部变量被第一次使用前，必须为其显式赋值

## Question 2

错误的回答:d

错误原因:

- `Overload` 类只有两个构造方法,其他两个是普通方法
## Question 3
程序输出:![image-20251018115623823](E:\a笔记\Markdown\Java\实验报告\Lab2\Lab2问答报告.assets\image-20251018115623823.png)
## Question 4

1. 能,理由如下:由输出

   

   ![image-20251018121420569](E:\a笔记\Markdown\Java\实验报告\Lab2\Lab2问答报告.assets\image-20251018121420569.png)

   ![image-20251018121434226](E:\a笔记\Markdown\Java\实验报告\Lab2\Lab2问答报告.assets\image-20251018121434226.png)

知,调用`b2 = new B(2);`与`b1 = new B(1);`时,B中的`A a6 = new A(6);` `A a7 = new A(a6);`与先开始执行,之后再执行`a8 = new A(8);`

2.能,理由如下:

由前述输出知
initialize A6 总是先于 copy from A6 输出

则`A a6 = new A(6);` 先执行,`A a7 = new A(a6); `后执行

## Question 5

- 在属性定义处显式初始化（如本例中的 `a1`）
- 在静态代码块中初始化（如本例中的 `a4`）

![image-20251018122011076](E:\a笔记\Markdown\Java\实验报告\Lab2\Lab2问答报告.assets\image-20251018122011076.png)

顺序:静态属性的初始化等同于它们在类定义中出现的顺序

## Question 6

1.不能

修改:
```java
public class Initialization {
    static {
        System.out.println("Initialization begin");
    }
    static B b1 = new B(1);
    static B b2;

    public static void main(String[] args) {
        System.out.println("main begins");
        A a9 = new A(9);
        b2 = new B(2);
        System.out.println("main ends");
    }
}
```

加上一个static块,发现输出如下
![image-20251018122447800](E:\a笔记\Markdown\Java\实验报告\Lab2\Lab2问答报告.assets\image-20251018122447800.png)

这说明在运行主类时B块与A块中static变量还没有被初始化,而是等到了被第一次被访问时才加载

2.

带有 static 关键字的方法、变量、代码块可调用带有 static 关键字的方法、变量

不带有static 关键字的方法、变量、代码块可调用任意有访问权限的变量

## Question 7

不能,因为唯一的构造函数带有private

## Question 8

因为唯一的构造函数带有private,导致只有`private static final Singleton uniqueInstance = new Singleton();`可以初始化对象

这个唯一的实例在类第一次被访问(类加载)时构造

## Question 9

`Singleton.getInstance().foo()`

## Question 10

| 修饰符           | 同一个类 | 同一个包 | 子类 | 所有类 |
| ---------------- | -------- | -------- | ---- | ------ |
| `private`        | 1        | 0        | 0    | 0      |
| `默认(无修饰符)` | 1        | 1        | 0    | 0      |
| `protected`      | 1        | 1        | 1    | 0      |
| `public`         | 1        | 1        | 1    | 1      |

### ## Question 11

```java
// Question 11
// 编写程序，在其中定义两个类

// Person 类：
// 属性有 name、age 和 sex；
// 提供你认为必要的构造方法；
// 方法 setAge() 设置人的合法年龄（0~130）；
// 方法 getAge() 返回人的年龄；
// 方法 work() 输出字符串 working；
// 方法 showAge() 输出 age 值。
// TestPerson 类：
// 创建 Person 类的对象，设置该对象的 name、age 和 sex 属性；
// 调用 setAge() 和 getAge() 方法，体会 Java 的封装性；
// 创建第二个对象，执行上述操作，体会同一个类的不同对象之间的关系。
class Person{
    String name;
    int age;
    String sex;
    Person(String name,int age,String sex){
        this.age=age;
        this.name=name;
        this.sex=sex.equals("male")||sex.equals("female")?sex :"male";
    }
    Person(){
        this.age=18;
        this.name="YXL";
        this.sex="male";
    }    
    public void setAge(int age){
        if(age>130) age=130;
        if(age<0) age=0;
        this.age=age;
    }
    public int getAge(){
        return age;
    }
    public void work(){
        System.out.println("working");
    }
    public void showAge(){
        System.out.println(""+age);
    }
}
public class TestPerson{
    public static void main(String[] args) {
        Person a=new Person();
        Person b= new Person("YYY",110,"male");
        a.setAge(100);
        a.showAge();
        b.setAge(123);
        b.showAge();
    }
}
```

## Question 12

```java
public class Question12 {
    static public boolean isNum(String num){
        if(num.charAt(0)=='0') return false;
        return num.matches("^[0-9]+$");
    }
    public static void main(String[] args) {
        String a="123321";
        System.out.println("Is num:"+isNum(a));
        if(!isNum(a)) return;
        System.out.println("Is Palindrome num:"+a.equals(new StringBuilder(a).reverse().toString()));
    }
}

```

