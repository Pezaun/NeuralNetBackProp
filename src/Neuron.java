
import java.util.Arrays;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 * Represents each neuron in a Multilayer Neural Net.
 * @author gabriel
 */
public class Neuron {

    RealMatrix weights;
    RealMatrix lastDeltaWeights;
    double sigError = 0.0;
    double sigOutput = 0.0;
    double wInputValue = 0.0;
    double targetOutput = -1;
    double[] input;

    /***
     * Instantiate a neuron that receives a number of synapses.
     * @param synapses define a number of synapses.
     */
    public Neuron(int synapses) {
        weights = new Array2DRowRealMatrix(NNMathUtils.weightAutoGenerate(synapses));
        lastDeltaWeights = new Array2DRowRealMatrix(new double[synapses]);
    }

    /***
     * Activates neuron with a given input.
     * @param input the input values for each synapse.
     */
    public void activate(double[] input) {
        this.input = input;
        wInputValue = NNMathUtils.thetaTrasnOfX(weights.getColumn(0), input);
        sigOutput = NNMathUtils.activation(wInputValue);
    }
    
    /***
     * Define classification error for a output layer neuron.
     */
    public void outputNeuronError() {
        sigError = (targetOutput - sigOutput) * (1 - sigOutput) * sigOutput;
    }

    /***
     * Define classification error for a hidden layer neuron.
     */
    public void hiddenNeuronError(int layer, int sinapse) {
        double nnSumation = 0;
        double[] nnWeights;
        double nnSigError = 0;
        for (int i = 0; i < NNBackProp.network[layer].length; i++) {
            nnWeights = NNBackProp.network[layer][i].weights.getColumn(0);
            nnSigError = NNBackProp.network[layer][i].sigError;
            nnSumation += nnWeights[sinapse] * nnSigError;
        }
        sigError = sigOutput * (1 - sigOutput) * nnSumation;
    }

    /***
     * Adjust synaptic weights for a hidden layer neurons.
     */
    public void adjustHiddenWeights() {
        double nw;
        double[] w = weights.getColumn(0);
        for (int i = 0; i < input.length; i++) {
            nw = w[i] + NNBackProp.learningRate * sigError * input[i];
            weights.setEntry(i, 0, nw);
        }
    }

    /***
     * Adjust synaptic weights for a output layer neurons.
     */
    public double adjustOutputWeights() {
        double nw;
        for (int i = 0; i < input.length; i++) {
            nw = weights.getEntry(i, 0) + NNBackProp.learningRate * sigError * input[i];
            weights.setEntry(i, 0, nw);
        }
        return sigError;
    }

    @Override
    public String toString() {
        StringBuilder out = new StringBuilder();
        out.append("\nClass Value: ");
        out.append(targetOutput);
        out.append("\nNeuron Weights: ");
        out.append(Arrays.toString(weights.getColumn(0)));
        out.append("\nInput Value: ");
        out.append(wInputValue);
        out.append("\nOutput Value: ");
        out.append(sigOutput);
        out.append("\nLocal Gradient(Error): ");
        out.append(sigError);
        return out.toString();
    }

}
