function [mat,avg1, prob2, diff2, prob3, diff3 ] = ttest_EEG(period)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
root_dir=['E:\脑电波\lz\全部数据\原始excel\平静\lz-调节2Hz_20160603_103047_ASIC_EEG.xlsx'];
list_file=dir(root_dir);
fileNum=size(list_file,1);
%period = 30;
for i=1:fileNum
    file_name = list_file(i).name;
%     [num,head,raw] = xlsread(['.\全部数据\原始excel\平静\' file_name],'第一组');
    [num,head,raw] = xlsread(['E:\脑电波\lz\全部数据\原始excel\平静\' file_name],'第一组');
    line = size(raw,1)-1;
    eeg = cell(ceil(line/period),9);
    index1= 2;
    lineNum=1;
    while index1+period <= line+1
        if isequal(raw(index1,22),raw(index1+period-1,22))
            for j=index1:index1+period
                for k=14:21
                    eeg{lineNum,k-13} = mean(num(index1:index1+period-1,k));
                end
            end
            eeg{lineNum,9} = num(index1,22);
            lineNum = lineNum + 1;
            index1 = index1 + ceil(period/4);
        else
            index1 = index1 + 1;
        end
    end
    
    mat = cell2mat(eeg);
    index = find(mat(:,9)==1);
    i1 = index(1);
    i2 = index(size(index,1));
    stage1 = mat(i1/2-15:i1/2+14,1:8);
    avg1 = mean(stage1);
    prob2 = ones(100,8);
    diff2 = ones(100,8);
    prob3 = ones(100,8);
    diff3 = ones(100,8);
    cnt = 1
    while i1+29 <= i2
        stage2 = mat(i1:i1+29,1:8);
        [h,p,ci,stats] = ttest2(stage1,stage2);
        prob2(cnt,:) = p;
        avg2 = mean(stage2);
        diff2(cnt,:) = (avg2-avg1)./avg1;
        i1 = i1 + ceil(30/4);
        cnt = cnt + 1;
    end
    i2 = i2 + 1;
    cnt = 1;
    while i2+29<lineNum
        stage3 = mat(i2:i2+29,1:8);
        [h,p,ci,stats] = ttest2(stage1,stage3);
        prob3(cnt,:) = p;
        avg3 = mean(stage3);
        diff3(cnt,:) = (avg3-avg1)./avg1;
        i2 = i2 + ceil(30/4);
        cnt = cnt + 1;
    end
end

end

