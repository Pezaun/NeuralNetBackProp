
import java.util.Arrays;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author gabriel
 */
public class Exec {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        NNBackProp nn = new NNBackProp(true,2,2,2,2,2,1);
        NNBackProp.learningRate = 1;
        nn.setSeedRandom(true);
        nn.setSeedValue(10);
        nn.setVerbose(true);
        nn.setWeights();
        double[] input = {0.35,0.9};
        
        nn.setTrainingValues(input, new double[]{0.5});
        
        nn.feedForward();        
        System.out.println(nn);
        
        System.out.println("----------------------------");   
        nn.backPropagate();
        System.out.println(nn);        
    }
    
}
