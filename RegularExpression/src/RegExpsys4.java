import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

//convert all DA first sentence file to systems1 input
public class RegExpsys4 {
    static String[] parts = new String[2];
    static HashMap<String, Integer> all_field_names = new HashMap<>();
    static HashMap<String, Integer> all_field_values = new HashMap<>();
    //static int train_setsize = 3834, valid_setsize = 352; //venue
    //static int train_setsize=4000,valid_setsize=455; //bio
    static int train_setsize = 42500, valid_setsize = 5310; //wikipedia


    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        File inputfile = new File("input.txt");

        try {
            PrintWriter output = new PrintWriter("input2.txt", "UTF-8");

            Scanner input = new Scanner(inputfile);

            String DA = new String(), summary = new String();

            while (input.hasNext()) {
                DA = input.nextLine();
                summary = input.nextLine();
                System.out.println("neee "+summary);
                if(!summary.contains("None-(Error: ")) {
                    output.println(DA);
                    output.println(summary);
                }
                else System.out.println("varmis");
            }

  output.close();
        } catch (IOException ex) {
        }
    }

}