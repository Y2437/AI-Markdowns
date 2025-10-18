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