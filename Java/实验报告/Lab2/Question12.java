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
