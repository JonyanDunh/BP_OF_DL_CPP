

#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <string>
#include <vector>
using namespace Eigen;
using namespace std;
class SigmoidLayer
{
    
public:
    MatrixXd out;
    MatrixXd forward(MatrixXd x) {
        MatrixXd re_x = -1 * x;
        out= 1 / (re_x.array().exp() + 1);
        return out;
    };
    MatrixXd backward(MatrixXd dout) {
        return dout.array() * (1.0 - out.array()) * out.array();

    };

};
class MulLayer {
public:
    MatrixXd x, y;
    MatrixXd forward(MatrixXd forward_x, MatrixXd forward_y) {
        x = forward_x;
        y = forward_y;
        return forward_x* forward_y;
    };
    MatrixXd* backward(MatrixXd dout) {
        static  MatrixXd result[] = { dout*y,dout*x };
        return result;
    };


};
class AddLayer
{
public:
    MatrixXd x, y;
    MatrixXd forward(MatrixXd forward_x, MatrixXd forward_y) {
        x = forward_x;
        y = forward_y;
        return forward_x + forward_y;
    };
    MatrixXd* backward(MatrixXd dout) {
        static  MatrixXd result[] = { dout,dout };
        return result;
    };


};
class ReLULayer {
public:
    MatrixXd mask;
    MatrixXd forward(MatrixXd x) {
        Matrix<bool, Dynamic, Dynamic> mask;
        mask = (x.array() <= 0);
        for (int i = 0; i < x.rows(); i++)
        {

            for (int z = 0; z < x.cols(); z++)
            {
                if (mask(i, z) == true)
                    x(i, z) = 0;
            }

        }
        return x;
    };
    MatrixXd backward(MatrixXd dout) {
        for (int i = 0; i < dout.rows(); i++)
        {

            for (int z = 0; z < dout.cols(); z++)
            {
                if (mask(i, z) == true)
                    dout(i, z) = 0;
            }

        }
        return dout;

    };

};
class SoftmaxLayer {
public:
    MatrixXd forward(MatrixXd x) {
        return x.array().exp()/ x.array().exp().sum();
    };



};
class Cross_entropy_error_Layer {
public:
    double forward(MatrixXd y, MatrixXd t) {
        double delta = 1e-7;
        return -(t.array() * (y.array() + delta).log()).sum();
    };


};
class AffineLayer {

public:
    MatrixXd W;
    MatrixXd b;
    MatrixXd x;
    MatrixXd dW;
    double db;
    void init(MatrixXd W1, MatrixXd b1) {

        W = W1;
        b = b1;
    }
    MatrixXd forward(MatrixXd x1) {
        x = x1;

        return x*W+b;
    }
    MatrixXd backward(MatrixXd dout) {
        MatrixXd dx = dout*W.transpose();
        dW = x.transpose() * dout;
        db = dout.sum();
        return dx;
    }
};
class SoftmaxWithLossLayer {
public:
    double loss;
    MatrixXd y;
    MatrixXd t;
    double forward(MatrixXd x, MatrixXd t1) {
        t = t1;
        SoftmaxLayer SoftmaxLayer;
        y = SoftmaxLayer.forward(x);
        Cross_entropy_error_Layer Cross_entropy_error_Layer;
        loss = Cross_entropy_error_Layer.forward(y, t);
        return loss;
    };
    MatrixXd backward() {
        int batch_size = t.rows();
        return (y - t).array() / batch_size;

    };

};
struct grads {
    MatrixXd W1;
    double b1;
    MatrixXd W2;
    double b2;
};
class Network_2_layer
{

public:
    map<string, MatrixXd> params;
    map<string, AffineLayer> Affinelayers;
    map<string, ReLULayer> ReLUlayers;
    SoftmaxWithLossLayer LastLayer;
    void init(int input_size, int hidden_size, int output_size, double weight_init_std = 0.01)
    {
        MatrixXd hidden_zeros(1, hidden_size);
        MatrixXd output_zeros(1, output_size);
        params["W1"] = weight_init_std * MatrixXd::Random(input_size, hidden_size);
        params["b1"] = hidden_zeros.setZero();
        params["W2"] = weight_init_std * MatrixXd::Random(hidden_size, output_size);
        params["b2"] = output_zeros.setZero();
        AffineLayer AffineLayer1;
        AffineLayer AffineLayer2;
        ReLULayer ReLULayer1;
        
        AffineLayer1.init(params["W1"], params["b1"]);
        AffineLayer1.init(params["W2"], params["b2"]);
        Affinelayers["Affine1"] = AffineLayer1;
        Affinelayers["Affine2"] = AffineLayer2;
        ReLUlayers["Relu1"] = ReLULayer1;
    }   

    MatrixXd  predict(MatrixXd x) {

        x=Affinelayers["Affine1"].forward(x);
        x = ReLUlayers["Relu1"].forward(x);
        x = Affinelayers["Affine2"].forward(x);
        return x;
        
    }
    double  loss(MatrixXd x, MatrixXd t) {
        
        MatrixXd y = predict(x);
        return LastLayer.forward(y, t);
    
    }
   /* double accuracy(MatrixXd x, MatrixXd t) {

        MatrixXd y = predict(x);



    }*/

    grads gradient(MatrixXd x, MatrixXd t)
    {
        loss(x, t);
        MatrixXd dout = LastLayer.backward();
        dout = Affinelayers["Affine2"].backward(dout);
        dout= ReLUlayers["Relu1"].backward(dout);
        dout = Affinelayers["Affine1"].backward(dout);
        grads grads;
        grads.W1 = Affinelayers["Affine1"].dW;
        grads.b1 = Affinelayers["Affine1"].db;
        grads.W2 = Affinelayers["Affine2"].dW;
        grads.b2 = Affinelayers["Affine2"].db;
        return grads;
    }
};
int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(string filename, vector<double>& labels)
{
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;


        for (int i = 0; i < number_of_images; i++)
        {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back((double)label);
        }

    }
}

void read_Mnist_Images(string filename, vector<vector<double>>& images)
{
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        unsigned char label;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
        cout << "rows = " << n_rows << endl;
        cout << "cols = " << n_cols << endl;

        for (int i = 0; i < number_of_images; i++)
        {
            vector<double>tp;
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char image = 0;
                    file.read((char*)&image, sizeof(image));
                    tp.push_back(image);
                }
            }
            images.push_back(tp);
        }
    }
}

int main()
{
    vector<double>labels;
    read_Mnist_Label("t10k-labels.idx1-ubyte", labels);
    vector<vector<double>>images;
    read_Mnist_Images("t10k-images.idx3-ubyte", images);


    auto m = images.size();      // 训练集矩阵行数
    auto n = images[0].size();   // 训练集矩阵列数
    auto b = labels.size();      // 训练集标签个数


     VectorXd actual_Y(b);                   // 初始化训练集实际值
     // 将训练集标签分类，0 = 1，非0 = -1
     for (unsigned i = 0; i < b; i++)
         labels[i] == 0 ? actual_Y(i) = 1 : actual_Y(i) = -1;

     Eigen::MatrixXd test = Eigen::Map<Eigen::Matrix<double, 3, 1> >(images.data());

     Network_2_layer network;
        network.init(784,50,10);
        //cout << Characteristic_matrix.cols();
        network.gradient(images, labels);

    return 0;
}

