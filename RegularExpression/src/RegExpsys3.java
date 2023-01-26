import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

//convert all DA first sentence file to systems1 input
public class RegExpsys3 {
    static String[] parts = new String[2];
    static HashMap<String, Integer> all_field_names = new HashMap<>();
    static HashMap<String, Integer> all_field_values = new HashMap<>();
    //static int train_setsize = 3834, valid_setsize = 352; //venue
    //static int train_setsize=4000,valid_setsize=455; //bio
    static int train_setsize = 42500, valid_setsize = 5310; //wikipedia



    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        File inputfile = new File("input.txt");

        try {
            PrintWriter output = new PrintWriter("allsummary.txt", "UTF-8");
            PrintWriter output2 = new PrintWriter("allbox.txt", "UTF-8");
            PrintWriter output3 = new PrintWriter("word_vocab.txt", "UTF-8");
            PrintWriter output4 = new PrintWriter("field_vocab.txt", "UTF-8");

            PrintWriter trainbox = new PrintWriter("train.box", "UTF-8");
            PrintWriter trainsummary = new PrintWriter("train.summary", "UTF-8");
            PrintWriter validbox = new PrintWriter("valid.box", "UTF-8");
            PrintWriter validsummary = new PrintWriter("valid.summary", "UTF-8");
            PrintWriter testbox = new PrintWriter("test.box", "UTF-8");
            PrintWriter testsummary = new PrintWriter("test.summary", "UTF-8");

            Scanner input = new Scanner(inputfile);

            String DA=new String(), summary = new String();

            while (input.hasNext()) {
                DA = input.nextLine();
                summary = input.nextLine();
                output2.println(convert_predicate(DA.substring(1,DA.length()-1)));
                parse_summary(summary);
                output.println(summary);
            }

           int limit=0;
           System.out.println("Toplam alan degeri sayisi " + all_field_values.size());
            for (Map.Entry<String, Integer> ent : sortByValue(all_field_values).entrySet()) {
                if(limit>=20000) break;
                if (ent.getKey().compareTo("none")!=0) {
                    output3.println(ent.getKey() + "\t" + ent.getValue());
                    limit++;
                }
            }

            System.out.println("Toplam alan isim sayisi " + all_field_names.size());
            System.out.println("Alan isimleri " + sortByValue(all_field_names));
            for (Map.Entry<String, Integer> ent : sortByValue(all_field_names).entrySet())
                if(ent.getValue()>=90)
                 output4.println(ent.getKey() + "\t" + ent.getValue());

            output.close();
            output2.close();
            output3.close();
            output4.close();

            Scanner inputbox = new Scanner(new File("allbox.txt"));
            Scanner inputsummary = new Scanner(new File("allsummary.txt"));
            int sentence_num=0;
            while (inputbox.hasNext()) {
                if(sentence_num<train_setsize){
                 trainbox.write(inputbox.nextLine()+"\n");
                 trainsummary.write(inputsummary.nextLine()+"\n");}
                else if((sentence_num>=train_setsize) && (sentence_num<train_setsize+valid_setsize)){
                    validbox.write(inputbox.nextLine()+"\n");
                    validsummary.write(inputsummary.nextLine()+"\n");
                }
                else
                {
                    testbox.write(inputbox.nextLine()+"\n");
                    testsummary.write(inputsummary.nextLine()+"\n");
                }
            sentence_num++;
            }

            trainbox.close(); trainsummary.close();
            validbox.close(); validsummary.close();
            testbox.close(); testsummary.close();
            //inputbox.close(); inputsummary.close();

        }

        catch(IOException ex){}
    }


    public static String convert_predicate(String text) {
        String ret_text = new String();
        String[] parts = new String[300];
        parts = text.split("@@@@");
        ret_text = "";
        for (int j = 0; j < parts.length; j++) {
            ret_text = ret_text + convert_field(parts[j]);
            if (j != parts.length - 1) ret_text = ret_text + "\t";
        }
        return ret_text;
    }


    public static String convert_field(String text) {
        String ret_text = new String();
        String field_name = new String();
        int counter = 1;
        String[] parts = new String[2];
        String[] field_values = new String[100];
        ret_text = "";
        parts = text.split(":");
        field_name = parts[0].substring(1,parts[0].length()-1);
        if (!all_field_names.containsKey(field_name))
            all_field_names.put(field_name, 1);
        else
            all_field_names.put(field_name, all_field_names.get(field_name) + 1);

        parts[1] = parts[1].replaceAll("([\\,])", " $1 ");
        parts[1] = parts[1].trim().replaceAll(" +", " ");
        System.out.println("parts"+parts[1]);
        field_values = parts[1].substring(1, parts[1].length() - 1).split(" ");

        for (int j = 0; j < field_values.length; j++) {
            ret_text = ret_text + field_name + "_" + Integer.toString(counter++) + ":" + field_values[j];

            if (!all_field_values.containsKey(field_values[j]))
                all_field_values.put(field_values[j], 1);
            else
                all_field_values.put(field_values[j], all_field_values.get(field_values[j]) + 1);

            if (j != field_values.length - 1) ret_text = ret_text + "\t";
        }
        return ret_text;
    }

    public static void parse_summary(String text) {
        String[] parts = new String[1000];
        parts = text.split(" ");
        for (int j = 0; j < parts.length; j++) {
            if (!all_field_values.containsKey(parts[j]))
                all_field_values.put(parts[j], 1);
            else
                all_field_values.put(parts[j], all_field_values.get(parts[j]) + 1);
        }
    }

    // function to sort hashmap by values
    public static HashMap<String, Integer> sortByValue(HashMap<String, Integer> hm) {
        // Create a list from elements of HashMap
        List<Map.Entry<String, Integer>> list =
                new LinkedList<Map.Entry<String, Integer>>(hm.entrySet());

        // Sort the list
        Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> o1,
                               Map.Entry<String, Integer> o2) {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        // put data from sorted list to hashmap
        HashMap<String, Integer> temp = new LinkedHashMap<String, Integer>();
        for (Map.Entry<String, Integer> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }
        return temp;
    }


  /*  //Uluc'un mekan dosyalari degismeli
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {

        File inputfile = new File("all-data.txt");
        try {
            PrintWriter output = new PrintWriter("alldata-corrected.txt", "UTF-8");

            Scanner input = new Scanner(inputfile);
            int turn =0;
            while (input.hasNext()) {
                String text = input.nextLine();
                if (text.contains("type =")) {
                   turn=1;
                   output.write(text+"\n");
                }
                else {
                    if(turn!=1)
                     output.write(text + "\n");
                    else{
                        if (text.compareTo("") == 0)
                            turn = 0;

                    }
                }

            }

            output.close();


        } catch (IOException ex) {
        }
    }*/

}