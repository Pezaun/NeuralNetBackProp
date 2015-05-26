
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
        NNBackProp nn = new NNBackProp(2,2,2,1);
        NNBackProp.learningRate = 1;
        NNBackProp.verbose = true;
        nn.setWeights();
        double[] input = {0.35,0.9};
        double[][][] weights = {{{1,1},{1,1}},{{0.1,0.8},{0.4,0.6}},{{0.3,0.9}}};
        nn.setWeights(weights);
        nn.setTrainingValues(input, new double[]{0.5});
        
        nn.feedForward();        
        System.out.println(nn);
        
        System.out.println("----------------------------");   
        nn.backPropagate();
        System.out.println(nn);        
    }
    
}
