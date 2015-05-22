
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Implements math utility operations.
 * @author gabriel
 */
public class NNMathUtils {
    
    /**
     * The activation function. This method depends of the Apache Commons Math 
     * library.
     * @param wiv the summation of the input values times weights.
     * @return 
     */
    static double activation(double wiv){
        Sigmoid sigmoid = new Sigmoid();
        return sigmoid.value(wiv);                    
    }
    
    /**
     * Multiply input values by weights. This method depends of the Apache Commons Math 
     * library.
     * @param w the weights.
     * @param a the input values.
     * @return 
     */
    static double thetaTrasnOfX(double[] w, double[] a){
        RealMatrix ww = new Array2DRowRealMatrix(w).transpose();
        RealMatrix aa = new Array2DRowRealMatrix(a);
        RealMatrix r = ww.multiply(aa);
        return r.getEntry(0, 0);
    }
    
    /**
     * Generate random value arrays
     * @param len the length of array to be generated.
     * @return a random filled array.
     */
    static double[] weightAutoGenerate(int len){
        double[] weights = new double[len];
        double max = 0.001;
        double min = -0.001;
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * (max - min)) + min;
        }
        return weights;
    }
    
    /**
     * Return the max value index of a given array.
     * @param values the max value index.
     * @return 
     */
    static int maxValueIndex(double[] values){
        int result = 0;
        double max = values[0];
        for (int i = 1; i < values.length; i++) {
            if(values[i] > max){
                max = values[i];
                result = i;
            }
        }
        return result;
    }
}