function [eeg] = get_GF( period )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    delta=2.5; %1-3Hz
    theta=6; %4-7Hz
    alpha1=9; %8-9Hz
    alpha2=11.5; %10-12Hz
    beta1=15.5;  %13-17Hz
    beta2=24.5;  %18-30Hz
    gamma1=36; %31-40Hz
    gamma2=46; %41-50Hz
    f=[2.5, 6, 9, 11.5, 15.5, 24.5, 36, 46];
    width=[3, 4, 2, 3, 5, 13, 10, 10];
    [num,head,raw] = xlsread(['E:\脑电波\lz\全部数据\原始excel\平静\cxq 早上2hz_20160728_111220_ASIC_EEG.xlsx'],'第一组');
    line = size(raw,1)-1;
    disp(line);
    eeg = cell(ceil(line/period),2);
    index1= 2;
    lineNum=1;
    while index1+period <= line+1
        fenzi=0.0;
        fenmu=0.0;
        for i=6:13
            fenzi = fenzi + mean(num(index1:index1+period-1,i))*f(1,i-5)*width(1,i-5);
            fenmu = fenmu + mean(num(index1:index1+period-1,i))*width(1,i-5);
        end
        eeg{lineNum,1} = fenzi/fenmu;
        eeg{lineNum,2} = num(index1,22);
        lineNum = lineNum + 1;
        index1 = index1 + ceil(period/4);
    end
  %  draw_baseLine(eeg);
end

