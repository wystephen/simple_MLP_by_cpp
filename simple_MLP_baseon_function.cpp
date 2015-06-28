#include <iostream>

#include <time.h>
#include <stdio.h>
#include<stdlib.h>
#include <math.h>

#include<omp.h>
#include "read_data.h"



#define random() double(double(rand()%100/1000.0)*10)


using namespace std;

double sigmod(double x){
    return 1.0/(1.0+exp(0.0-x));
}
double dsigmoid(double x){
    return x*(1.0-x);
}
//初始化权值的函数
void weight_random(double *w,int out_size,int in_size){
//#pragma omp parallel for
    for(int i = 0;i<out_size;i++){
        for (int j = 0;j<in_size;j++){
            *(w+i*in_size+j)=random();
            *(w+i*in_size+j)=*(w+i*in_size+j)/4.0-0.1;

            //cout <<i<<"--"<<j<<endl;
            //cout<<"w" <<*(w+i*in_size+j)<<endl;
        }
    }
    //int k=0;
    //cin>>k;
}
//根据权值计算下一层值
void get_out(double *in,double *out,double *w,int out_size,int in_size){
//#pragma omp parallel for
    for(int i = 0;i<out_size;i++){
        *(out+i)=0;
        for (int j = 0;j<in_size;j++){
            *(out+i)+=(*(w+i*in_size+j))*(*(in+j));
            //cout <<i<<"--"<<j<<endl;
            //cout <<"w["<<i<<","<<j<<"]:"<<*(w+j*out_size+i)<<endl;
        }
        *(out+i)=sigmod(*(out+i));
        int k=0;
        //cin>>k;
        //cout<<"out["<<i<<"]:"<<*(out+i)<<endl;
    }
}

void get_err_out(double *out,double *err_out,double *rel_out,int out_size){
    for(int i(0);i<out_size;i++){
        *(err_out+i)=dsigmoid(*(out+i))*(*(rel_out+i)-(*(out+i)));
        //cout <<"err_out["<<i<<"]:"<< *(err_out+i)<<"rel_out:"<<*(rel_out+i)<<" cha out:"<<(*(rel_out+i)-(*(out+i)))<<endl;
    }

}
void get_err_hidden(double *hidden,double *err_hidden,double *err_out,double *w,int out_size,int in_size){
    for(int i(0);i<in_size;i++){
        *(err_hidden+i)=0.0;
        double sum(0);
        for(int j(0);j<out_size;j++){
            sum+=(*(w+j*in_size+i))*(*(err_out+j));
        }
        *(err_hidden+i)=dsigmoid((*(hidden+i)))*(sum);
        //cout << "err_hidden["<<i<<"]:"<<*(err_hidden+i)<<endl;
    }
}
void updata_weight(double *w,double *err_out,double *in,double learning_rate,int out_size,int in_size){

    for(int i(0);i<in_size;i++){
        for(int j(0);j<out_size;j++){
            *(w+j*in_size+i)+=learning_rate * (*(in+i))*(*(err_out+j));

        }
    }
}


int main()
{
    double *x;
    double *y;
    x = read_data_x("D:\\project\\MLP\\images.txt");
    y = read_data_y("D:\\project\\MLP\\lables.txt");
    cout << "读取数据完毕！"<<endl;
    cout <<"end read data!"<<endl;
    srand(time(0));
    //double x[4][2]={{1,1},{1,0},{0,1},{0,0}};
    //double y[4]={0,1,1,0};
    //以上为亦或
    int IN_size,OUT_size,HIDDEN_size;
    int data_num=get_num_of_dataset("D:\\project\\MLP\\lables.txt");
    //cout << "getlen(x):"<<getlen(x)<<"getlen(y):"<<getlen(y)<<endl;

    int x_size = get_size("D:\\project\\MLP\\images.txt");
    int y_size = get_size("D:\\project\\MLP\\lables.txt");
    IN_size=int(x_size/data_num);
    OUT_size =int(y_size*10/data_num);
    HIDDEN_size = int(double((IN_size+OUT_size))/2);
    cout<<"OUT:"<<OUT_size<<"IN:"<<IN_size<<endl;

    double *w1=new double[HIDDEN_size*IN_size];
    double *w2=new double[OUT_size*HIDDEN_size+1];


    double old_w1[HIDDEN_size*IN_size];
    double old_w2[OUT_size*HIDDEN_size+1];

    double delta_w1[HIDDEN_size*IN_size];
    double delta_w2[OUT_size*HIDDEN_size+1];

    double err_out[OUT_size];
    double err_hidden[HIDDEN_size];

    double HIDDEN[HIDDEN_size+1];
    double OUT[OUT_size];

    HIDDEN[HIDDEN_size]=1.0;

    std::cout <<"Begin to random weight！"<<endl;
    weight_random(w1,HIDDEN_size,IN_size);
    weight_random(w2,OUT_size,HIDDEN_size+1);
    std::cout <<"End pre-random weight!"<<endl;


    for(int epoch = 0;epoch<100000000;epoch++)
    {
        if (epoch%500==0){
            std::cout << epoch<<endl;
            }

        for(int index = 0;index <data_num;index++){
            //训练部分
            double IN[IN_size];
            for(int i(0);i<IN_size;i++){
                IN[i]=*(x+index*IN_size+i);
            }
            //cout <<IN[0]<<" "<<IN[1]<<endl;
            get_out(IN,HIDDEN,w1,HIDDEN_size,IN_size);

            get_out(HIDDEN,OUT,w2,OUT_size,HIDDEN_size+1);
            double rel_out[OUT_size];
            //rel_out = y[index];
            //cout << OUT[0] ;
            //cout <<"OUT size:"<<OUT_size<<endl;
            for(int k(0);k<OUT_size;k++){
               // cout <<"OUT"<<*(y+index*OUT_size+k)<<endl;
                *(rel_out+k)=*(y+index*OUT_size+k);
            }
            get_err_out(OUT,err_out,rel_out,OUT_size);
            get_err_hidden(HIDDEN,err_hidden,err_out,w2,OUT_size,HIDDEN_size+1);

            updata_weight(w2,err_out,HIDDEN,0.5,OUT_size,HIDDEN_size+1);
            updata_weight(w1,err_hidden,IN,0.5,HIDDEN_size,IN_size);

            int k=0;
            //k=getchar();

            if (epoch%10000==0){
               // cout<<"rel_out:"<<rel_out<<"OUT:"<<OUT[0]<<endl;
                double err(0.0);
                for(int k(0);k<OUT_size;k++){
                    err+=abs(double(*(rel_out+k))-(*(OUT+k)));
                    if(*(rel_out+k)==1){
                        cout<<"rel_out:"<<k;
                    }
                    if(*(OUT+k)>0.5){
                        cout<<"OUT:"<<k<<"  ";
                    }
                }
                cout <<" --"<<*(rel_out+k)<<"---"<<*(OUT+k)<<endl;
                cout << "err:"<<err<<endl;

            }

        }


    }


    return 0;
}

