%tester_name,activity,position;
%function transform_format(tester_name,activity,position);
root_dir=['E:\脑电波\lz\全部数据\原始txt\AslcData\平静\赵师弟-调节2hz_20160728_105157_ASIC_EEG.txt'];
list_txt=dir(root_dir);
Fs = 30;
size(list_txt)
for i=1:size(list_txt,1)
    file_name =  list_txt(i).name;
    dataset_im = importdata(['E:\脑电波\lz\全部数据\原始txt\AslcData\平静\' file_name],':');
    acc =dataset_im;
    prefix=regexp(file_name, '\.', 'split');
    deststr=['E:\脑电波\lz\全部数据\原始excel\平静\' prefix{1} '.xlsx'];
    data1=cell(18000,9);
    
    head =[{'编号'},{'Poor Signal'},{'Attention eSense'},{'Meditation eSense'},{'Blink Strength'},{'Delta(1-3Hz)'},{'Theta(4-7Hz)'},{'Low Alpha(8-9Hz)'},{'High Alpha(10-12Hz)'},{'Low Beta(13-17Hz)'},{'High Beta(18-30Hz)'},{'Low Gamma(31-40Hz)'},{'Mid Gamma(41-50Hz)'},{'Delta所占百分比'},{'Theta所占百分比'},{'Low Alpha所占百分比'},{'High Alpha所占百分比'},{'Low Beta所占百分比'},{'High Beta所占百分比'},{'Low Gamma所占百分比'},{'Mid Gamma所占百分比'},{'是否调节'},{'时间'}];
    data1(1,1:size(head,2))=head(:);
    line = 1;
    line = line + size(acc.textdata,1);
    index = 2;
    for j=1:size(acc.textdata,1)
        if abs(str2double(acc.textdata{j,2})-0) < eps
            total = 0.0;
            for k =1:13
                data1{index,k}=str2double(acc.textdata{j,k});
                if k>5
                    total = total + data1{index,k};
                end
            end
            data1{index,14} = data1{index,6}/total;
            data1{index,15} = data1{index,7}/total;
            data1{index,16} = data1{index,8}/total;
            data1{index,17} = data1{index,9}/total;
            data1{index,18} = data1{index,10}/total;
            data1{index,19} = data1{index,11}/total;
            data1{index,20} = data1{index,12}/total;
            data1{index,21} = data1{index,13}/total;
            if strcmpi(acc.textdata{j,14},'true')
                data1{index,22} = 1;
            else
                data1{index,22} = 0;
            end
            data1{index,23} = acc.data(j,1);
            index = index + 1;
        end
    end
    %         for m= 14:21
    %             total1 = 0.0;
    %             num = size(acc.textdata,1)+1;
    %             for n=2:num
    %                 total1 = total1 + cell2mat(data1(n,m));
    %             end
    %             data1(num+1,m) = num2cell(total1/size(acc.textdata,1));
    %         end
    %line = line + 10;
    if(exist(deststr,'file'))
       delete(deststr);
    end
    xlswrite(deststr,data1(1:line,:),'第一组',['A1:W' num2str(line)]);
end