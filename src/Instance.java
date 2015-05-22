/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Classe representa instâncias de teste e treinamento.
 * @author gabriel
 */
public class Instance {
    int getFeaturesLength(){
        return featureValues.length;
    }
    
    int getOutputLength(){
        return output.length;
    }
    double[] featureValues;
    double[] output;
}
