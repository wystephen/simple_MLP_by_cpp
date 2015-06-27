#include <iostream>

#include <time.h>
#include <stdio.h>
#include<stdlib.h>
#include <math.h>

#include<omp.h>

#define random() double(double(rand()%100/1000.0)*10)


using namespace std;

double sigmod(double x){
    return 1.0/(1.0+exp(0.0-x));
}
double dsigmoid(double x){
    return x*(1.0-x);
}

void weight_random(double *w,int out_size,int in_size){
//#pragma omp parallel for
    for(int i = 0;i<out_size;i++){
        for (int j = 0;j<in_size;j++){
            *(w+i*in_size+j)=random();
            *(w+i*in_size+j)=*(w+i*in_size+j)/4.0-0.1;

            //cout <<i<<"--"<<j<<endl;
            cout<<"w" <<*(w+i*in_size+j)<<endl;
        }
    }
    //int k=0;
    //cin>>k;
}
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
{   srand(time(0));
    double x[4][2]={{1,1},{1,0},{0,1},{0,0}};
    double y[4]={0,1,1,0};
    //以上为亦或
    int IN_size,OUT_size,HIDDEN_size;
    IN_size=2;
    OUT_size = 1;
    HIDDEN_size = 8;

    double w1[HIDDEN_size*IN_size];
    double w2[OUT_size*HIDDEN_size+1];


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


    for(int epoch = 0;epoch<100000;epoch++)
    {
        if (epoch%50==0){
            std::cout << epoch<<endl;
            }

        for(int index = 0;index < 4;index++){
            //训练部分
            double IN[IN_size];
            for(int i(0);i<IN_size;i++){
                IN[i]=x[index][i];
            }
            //cout <<IN[0]<<" "<<IN[1]<<endl;
            get_out(IN,HIDDEN,w1,HIDDEN_size,IN_size);

            get_out(HIDDEN,OUT,w2,OUT_size,HIDDEN_size+1);
            double rel_out;
            rel_out = y[index];
            //cout << OUT[0] ;
            get_err_out(OUT,err_out,&rel_out,OUT_size);
            get_err_hidden(HIDDEN,err_hidden,err_out,w2,OUT_size,HIDDEN_size+1);

            updata_weight(w2,err_out,HIDDEN,0.5,OUT_size,HIDDEN_size+1);
            updata_weight(w1,err_hidden,IN,0.5,HIDDEN_size,IN_size);

            int k=0;
            //k=getchar();



        }

    }


    return 0;
}

