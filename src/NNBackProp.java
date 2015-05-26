
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 * Represents a Neural Network instance.
 * @author gabriel
 */
public class NNBackProp {
    static boolean verbose = false;
    double[] inputValues;
    double[] classValues;
    static Neuron[][] network;
    static double learningRate = 0;
    static double limitError = 0.1;
    static int epoch = 1;
    int[] architecture;
    int epochCount = 0;
    Instance[] instances;
    NNObserver observer;

    /**
     * Define a interface that observe the network training process.
     * @param observer a NNObserver implementation to be notified.
     */
    public void setObserver(NNObserver observer) {
        this.observer = observer;
    }
    
    /**
     * Define epoch number for training.
     * @param epoch the epoch number.
     */
    public static void setEpoch(int epoch) {
        NNBackProp.epoch = epoch;
    }

    /**
     * Instantiate a new network with a given architecture.
     * @param architecture an integer array with a number of neurons in each layer.
     */
    public NNBackProp(int... architecture) {
        this.architecture = architecture;
        network = new Neuron[architecture.length - 1][];
        for (int i = 1; i < architecture.length; i++) {//COL
            network[i - 1] = new Neuron[architecture[i]];
            for (int j = 0; j < architecture[i]; j++) {//ROW
                network[i - 1][j] = new Neuron(architecture[i - 1]);
            }
        }
    }

    public static void setLimitError(double limitError) {
        NNBackProp.limitError = limitError;
    }

    public static void setLearningRate(double learningRate) {
        NNBackProp.learningRate = learningRate;
    }

    public void setTrainingValues(double[] values, double[] classValue) {
        inputValues = values;
        this.classValues = classValue;
        for (int i = 0; i < network[network.length - 1].length; i++) {
            network[network.length - 1][i].targetOutput = classValue[i];
        }
    }

    public void setTrainingValues(Instance instance) {
        setTrainingValues(instance.featureValues, instance.output);
    }

    public void setTrainingValues(List<Instance> instance) {
        this.instances = new Instance[instance.size()];
        int i = 0;
        for (Instance inst : instance) {
            this.instances[i++] = inst;
        }
    }

    public void setWeights(double[][][] weights) {
        for (int i = 0; i < network.length; i++) {
            for (int j = 0; j < network[i].length; j++) {
                network[i][j].weights = new Array2DRowRealMatrix(weights[i][j]);
            }
        }
    }

    public void setWeights() {
        for (int i = 0; i < network.length; i++) {
            for (Neuron network1 : network[i]) {
                if (i == 0) {
                    network1.weights = new Array2DRowRealMatrix(
                            NNMathUtils.weightAutoGenerate(architecture[0]));
                } else {
                    network1.weights = new Array2DRowRealMatrix(
                            NNMathUtils.weightAutoGenerate(network[i - 1].length));
                }
            }
        }
    }

    /**
     * A feed forward training execution.
     */
    void feedForward() {
        double[] layerValues;
        for (int i = 0; i <= network.length - 1; i++) {
            if (i == 0) {
                for (Neuron hiddenLayer1 : network[0]) {
                    hiddenLayer1.activate(inputValues);
                }
                continue;
            }
            layerValues = new double[network[i - 1].length];
            for (int j = 0; j < network[i - 1].length; j++) {
                layerValues[j] = network[i - 1][j].sigOutput;
            }

            for (Neuron networkLayer : network[i]) {
                networkLayer.activate(layerValues);
            }
        }
    }

    /**
     * A back-propagation training execution.
     */
    void backPropagate() {
        for (int i = 0; i < network[network.length - 1].length; i++) {
            network[network.length - 1][i].outputNeuronError();
        }
        for (int i = 0; i < network[network.length - 1].length; i++) {
            network[network.length - 1][i].adjustOutputWeights();
        }

        for (int i = architecture.length - 2; i > 0; i--) {
            for (int j = 0; j < architecture[i]; j++) {
                network[i - 1][j].hiddenNeuronError(i, j);
            }
            for (int j = 0; j < architecture[i]; j++) {
                network[i - 1][j].adjustHiddenWeights();
            }
        }
    }

    /**
     * Get entire output for a network.
     * @return an arrays with the output value of each output layer neuron.
     */
    double[] getOutput() {
        double[] r = new double[architecture[architecture.length - 1]];
        for (int i = 0; i < r.length; i++) {
            r[i] = network[architecture.length - 2][i].sigOutput;
        }
        return r;
    }

    /**
     * Execute a train session by a given epoch number.
     * @param verbose if true, print training debug messages.
     */
    void train(boolean verbose) {
        epochCount = 0;
        int instanceCount = 1;
        for (int epoch = 1; epoch <= this.epoch; epoch++) {
            instanceCount = 1;
            for (Instance inst : instances) {
                if (verbose) {
                    System.out.printf("Training instance %d on the epoch %d\n", instanceCount++, epoch);
                }
                setTrainingValues(inst.featureValues, inst.output);
                feedForward();
                backPropagate();
            }
            if (verbose) {
                System.out.println();
            }
            epochCount++;
            if(observer != null){
                observer.notifyObserver();
            }
            Collections.shuffle(Arrays.asList(instances));
        }        
    }

    /**
     * Get the summation of all output layer neurons error.
     * @return the summation of erros.
     */
    double getOutputError() {
        double error = 0;
        for (Neuron n : network[architecture.length - 2]) {
            error += Math.abs(n.targetOutput - n.sigOutput);
        }
        return error;
    }

    @Override
    public String toString() {
        StringBuilder out = new StringBuilder();
        if (verbose) {
            out.append("Input Layer:\n");
            for (int i = 0; i < inputValues.length; i++) {
                out.append(i + 1);
                out.append(" Value: ");
                out.append(inputValues[i]);
                out.append("\n");
            }
            for (int i = 0; i < network.length; i++) {
                out.append("\nLayer ");
                out.append(i + 1);
                out.append(":\n");
                for (Neuron network1 : network[i]) {
                    out.append(network1.toString());
                    out.append("\n");
                }
            }
            return out.toString();
        } else {
            out.append("\nOutput Layer ");
            out.append(":\n");
            for (Neuron network1 : network[network.length - 1]) {
                out.append(network1.toString());
                out.append("\n");
            }
            out.append("\n#########################");
            return out.toString();
        }

    }

}
