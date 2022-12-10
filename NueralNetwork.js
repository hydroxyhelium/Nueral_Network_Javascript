import { JSarray, CreateZeroArray } from "./JavaScript-LinearAlgebra-library/JSarray"


class Module{
    constructor(){ 
    }
    forward(jsarray){}
    backward(jsarray){}
    sgd_step(lrate){}
}

// Each column of W would represent the weights attached to a particular node of the nueral network. 
// Dimension of W would be (input_dim, output_dimesnion)
// Dimensiono of W0 would be (input_dim, 1) 

class NueralNetwork extends Module{

    constructor(input_size,output_size){
        this.input_size = input_size
        this.output_size = output_size
        this.W = CreateZeroArray([input_size, output_size])
        this.W0 = CreateZeroArray([output_size, 1])
    }


    // input array into a feed-forward nueral network would be of the size (input_size,1)
    // Output array would be of the form (output_size, 1)

    forward(A){
        self.A = A
        
        return (self.W.matrixproduct(A)).add(self.W0)
    }

    // Assumes that backward function recieves dLdZ (partial derivative of the loss function w.r.t linear product)
    // returns dLdA (partial derivative of the loss function w.r.t output from activation function module)

    // While doing so it saves dLdW, dLdW0

    // 
    backward(dLdZ){
        self.dLdW = (self.A).matrixproduct(dLdZ.transpose())
        self.dLdW0 = dLdZ

        return (self.W.transpose()).matrixproduct(dLdZ)
    }

    // lrate is an integer
    sgd(lrate){
        self.W = self.W.add((self.dLdW).multiply(-1*lrate))
        self.W0 = self.W0.add((self.dLdW0).multiply(-1*lrate))
    }
}

class Relu extends Module{
    constructor(input_size){
        this.input_size = input_size
    }

    // forward recieves linear combination of the previous layer
    // It then applies the function to each of the numbers and returns it
    forward(Z){
        self.Z = Z 
        self.output = CreateZeroArray([this.input_size, 1])
        for(var i=0; i<this.input_size; ++i){
            self.output.getelement(i).getelement(0) = Z.getelement(i).getelement(0)<0? 0 : Z.getelement(i).getelement(0)         
        }

        return self.output
    }

    // We need to return dLdZ
    // uses stored Z to calculate dAdZ 
    backward(dLdA){
        var dAdZ = CreateZeroArray([this.Z.array.length,1]) // initial value 

        for(var i=0; i<this.Z.array.length; ++i){
            dAdZ.getelement(i).getelement(0) = this.Z.getelement(i).getelement(0) <= 0 ? 0 : 1 
        }

        return dLdA.elementproduct(dAdZ)
    }
}


class Sigmoid extends Module{

    // this returns A.
    // Z needs to be stored for the backwards pass.  

    forward(Z){
        this.Z = Z 
        this.output = CreateZeroArray([this.Z.array.length, 1]) // a temporary array of 0's 


        for(var i=0; i<this.Z.array.length; ++i){
            this.output.getelement(i).getelement(0)= 1/(1+Math.E**(-1*(this.Z.getelement(i).getelement(0))))
        }

        return this.output
    }

    // this returns dLdZ 
    backward(dLdA){
        var dAdZ = CreateZeroArray([this.output.array.length, 1])

        for(var i=0; i< this.output.array.length;++i){
            dAdZ.getelement(i).getelement(0) = 1 - this.output.getelement(i).getelement(0)**2
        }

        return dLdA.elementproduct(dAdZ)
    }
}


class Softmax extends Module{

    constructor(input_size){
        this.input_size = input_size
    }


    forward(Z){
        this.output = CreateZeroArray([this.input_size, 1])
        sum = 0 
        for(var i=0; i<this.input_size; ++i){
            sum += Math.E**(Z.getelement(i).getelement(0))
            this.output.getelement(i).getelement(0) = Math.E**(Z.getelement(i).getelement(0))
        }
        
        this.output = this.output.multiply(1/(sum))

        return this.output
    }


    // We handle this case in the loss function modue by directly returning dLdZ instead of dLdA
    backward(dLdZ){
        return dLdZ    
    }

}


// negative log likelyhood loss function
class NLL extends Module{

    // cacluslates the loss based on predictions and actual values
    // loss =  - y * log(y_pred)
    // returns a scalar value

    constructor(input_size){
        this.input_size = input_size
    }

    forward(ypred, y){
        this.output = 0 
        this.ypred = ypred
        this.y = y 

        for(var i =0; i<this.input_size; ++i){ 
            this.output += -1*(y.getelement(i).getelement(0)*(Math.log(ypred.getelement(i).getelement(0))))   
        }

        return this.output
    }

    // uses the stored y_pred, y to return dLdZ 
    // ie. partial derivative of the loss function (NLL) w.r.t to the linear output of the last layer. Ie. the pre-activation values
    backward(){
        return this.ypred.add(this.y.multiply(-1))
    }
}


// SGD is used to train this model
class Sequential{

    // modules is a list of NueralNetwork|RELU|Sigmoid|Softmax
    // loss is a loss function that is used to evaluate the predictions. 
    constructor(modules, loss){
        this.modules = modules
        this.loss = loss
        this.l = modules.length
    }

    sgd(X, y){



    }

    // forward pass
    forward(Xt){
        // assumes that Xt has dim. (X, 1)

        var temp = Xt
        for(var i=0; i<this.l; ++i){
            temp = this.modules[i].forward(temp)
        }
    
        return temp
    }

    // backward pass
    backward(gamma){
        var temp = gamma
        for(var i=this.l-1; i>-0; --i){
            temp = this.modules[i].backward(temp)
        }
    }

    // performances gradient descent step on all the modules, assumes that
    // backward pass has already been performed. 
    sgd_step(lrate){
        for(var i=0; i<this.l; ++i){
            this.modules[i].sgd_step(lrate)
        }
    }

}
