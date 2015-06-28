#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;


double *read_data_x(string file_name){
    ifstream inf;
    inf.open(file_name.c_str());
    vector<double> x_tmp;
    double ll;
    while(inf>>ll){
       x_tmp.push_back(ll);
    }
    cout << x_tmp.size()<<endl;
    double *x = new double[x_tmp.size()];
    for(int i(0);i<x_tmp.size();i++){
        *(x+i)=x_tmp.at(i)/250.0;
    }
    return x;
}
int get_size(string file_name){
    ifstream inf;
    inf.open(file_name.c_str());
    vector<double> x_tmp;
    double ll;
    while(inf>>ll){
       x_tmp.push_back(ll);
    }
    return x_tmp.size();
}

double *read_data_y(string file_name){
    ifstream inf;
    inf.open(file_name.c_str());
    vector<double> x_tmp;
    double ll;
    while(inf>>ll){
       x_tmp.push_back(ll);
    }
    cout << x_tmp.size()<<endl;
    double *x = new double[x_tmp.size()*10];
    for(int i(0);i<x_tmp.size();i++){
        for(int j(0);j<10;j++){
            if(j==int(x_tmp.at(i))){
                *(x+i*10+j)=1.0;
            }else{
                *(x+i*10+j)=0.0;
            }
        }

    }
    return x;
}

int get_num_of_dataset(string file_name){
    ifstream inf;
    inf.open(file_name.c_str());
    vector<double> x_tmp;
    double ll;
    while(inf>>ll){
       x_tmp.push_back(ll);
    }
    cout <<"here"<<endl;
    cout <<x_tmp.size()<<endl;
    return x_tmp.size();
}
