#include <fstream>
#include <iostream>
#include <random>
#include "../net/include/Activation.hpp"
#include "../net/include/Loss.hpp"
#include "../net/include/Initialisers.hpp"
#include "../net/include/DenseLayer.hpp"
#include "../net/include/Network.hpp"

using Vec = std::vector<double>;


std::vector<Vec> grid(int n=200){
    std::vector<Vec> g; g.reserve(n*n);
    double step=1.0/(n-1);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) g.push_back({i*step,j*step});
    return g;
}

int main(){
    Network net;
    auto& l1 = net.add<DenseLayer>(2,3, act::tanh, act::tanh_prime);
    auto& l2 = net.add<DenseLayer>(3,1, act::sigmoid, act::sigmoid_prime);

    std::mt19937 gen(std::random_device{}());
    xavier(l1.W, l1.b, 2, 3, 2025);
    xavier(l2.W, l2.b, 3, 1, 2025);

    std::vector<Vec> X{{0,0},{0,1},{1,0},{1,1}};
    std::vector<Vec> Y{{0},{1},{1},{0}};

    double lr = 0.01;
    int batch = 1'000'000;

    for(int e=1; e<= batch; ++e){
        double loss_sum=0;
        for(size_t i=0;i<X.size();++i){
            auto& x = X[i]; auto& y = Y[i];
            const Vec& o = net.forward(x);
            loss_sum += loss::binary_cross_entropy(o,y);
            net.backward(x, loss::bce_grad(o,y), lr);
        }
        if(e<=10||e%1000==0) std::cout<<"Epoch "<<e<<" loss "<<loss_sum/X.size()<<"\n";
    }

    // CSV for contour plotting
    std::ofstream csv("xor_grid.csv");
    csv<<"x,y,prob\n";
    for(auto& p:grid()) csv<<p[0]<<','<<p[1]<<','<<net.forward(p)[0]<<'\n';
}
