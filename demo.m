% Demo for paper "Saliency Detection via Graph-Based Manifold Ranking" 
% by Chuan Yang, Lihe Zhang, Huchuan Lu, Ming-Hsuan Yang, and Xiang Ruan
% To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.


addpath('./others/');
addpath('./slic/');       %进行超像素分割的函数
%%------------------------set parameters---------------------%%
theta=10; % control the edge weight  控制边缘权重
alpha=0.01;% control the balance of two items in manifold ranking cost function  控制流形排序成本函数中两项的平衡
% spnumber=3000;% superpixel number  超像素数量
imgRoot='./test/';% test image path
saldir='./saliencymap/';% the output path of the saliency map
supdir='./superpixels/';% the superpixel label file path
foredir='./foreground/';%the foreground map
mkdir(supdir);  %指定的文件夹下创建新的文件夹
mkdir(saldir);
mkdir(foredir);
% imnames=dir([imgRoot '*' 'jpg']);  %列出当前目录下符合的文件夹和文件

%% 读取数据并构造对数比差异图
fid=fopen('C:\Users\dell\Desktop\Cd\20120111', 'r');
[data,~]=fread(fid,[2240*2,2420],'float'); 
a1=data(1:2:end,:);
b1=data(2:2:end,:);
c1=a1+i*b1;
amli=abs(c1);

fid=fopen('C:\Users\dell\Desktop\Cd\20130119', 'r');
[data2,count]=fread(fid,[2240*2,2420],'float'); 
a2=data2(1:2:end,:);
b2=data2(2:2:end,:);
c2=a2+i*b2;
amli2=abs(c2);

w = fspecial('gaussian',[9,9],5);
amli = imfilter(amli,w,'replicate'); %高斯滤波
amli2 = imfilter(amli2,w,'replicate'); %高斯滤波
[r,c]=size(amli);
 div_log=zeros(r,c);  %提前分配空间

 for i=1:r
    for k=1:c
        div_log(i,k)=abs(log(amli2(i,k)+1)-log(amli(i,k)+1));   %对图像矩阵进行遍历计算
    end
 end
% div_log_min=min(min(div_log),[],2);
% div_log_max=max(max(div_log),[],2);
 div_mean=3*mean(mean(div_log));
 for i=1:r
    for k=1:c
%         div_log(i,k)=(div_log(i,k)-div_log_min)/(div_log_max-div_log_min);    %进行归一化
        if div_log(i,k)<div_mean
            div_log(i,k)=0;
        end
    end
 end
div_log=medfilt2(div_log,[11,11]);%中值滤波
figure
imshow(div_log);
%% 进行超像素分割
%  [D, Am, C,h] = slic(A, 2500,10, 1.5);
 [superpixels] = slic(div_log, 1500,10);     
 I=drawregionboundaries(superpixels, div_log, [255 0 0]);
 figure
 imshow(I);
%%
% for ii=1:length(imnames)   
%     disp(ii);
%     imname=[imgRoot imnames(ii).name]; 
%     div_log=imread(imname);
%     [input_im,w]=removeframe(imname);% run a pre-processing to remove the image frame 运行预处理来删除图像边框
    [m,n] = size(div_log);
    w=[m,n,1,m,1,n];
    
    
%%----------------------generate superpixels--------------------%%
%     imname=[imname(1:end-4) '.bmp'];% the slic software support only the '.bmp' image
%     comm=['SLICSuperpixelSegmentation' ' ' imname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];
%     system(comm);    
%     spname=[supdir imnames(ii).name(1:end-4)  '.dat'];
%     superpixels=ReadDAT([m,n],spname); % superpixels就是超像素分割的结果，一个label矩阵，数字代表该像素的标签或者类别。
%     [superpixels] = slic(div_log, 1500,30);     
%     I=drawregionboundaries(superpixels, div_log, [255 0 0]);
%     imshow(I);

%%----------------------design the graph model--------------------------%%
% compute the feature (mean color in lab color space)  计算特征(实验室颜色空间中的平均颜色)
% for each node (superpixels)
    spnum=max(superpixels(:));% the actual superpixel number  实际的超像素数量
    input_vals=reshape(div_log, m*n, 1); %将input_im排列成m*n行，k列的矩阵
    gray_vals=zeros(spnum,1);   %spnum是超像素数量
    inds=cell(spnum,1);  %生成元组
    for i=1:spnum
        inds{i}=find(superpixels==i);  %存储相同标签的像素的位置
        gray_vals(i,1)=mean(input_vals(inds{i},:),1);
    end  
    seg_vals=gray_vals;
%    lab_vals = colorspace('Lab<-', gray_vals);  %将rgb转为Lab
%     seg_vals=reshape(lab_vals,spnum,3);% feature for each superpixel
 
 % get edges
    adjloop=AdjcProcloop(superpixels,spnum);
    edges=[];
    for i=1:spnum
        indext=[];
        ind=find(adjloop(i,:)==1);   %找到与第i个超像素邻接的周围超像素，返回位置
        for j=1:length(ind)
            indj=find(adjloop(ind(j),:)==1);  %邻接的邻接的像素
            indext=[indext,indj];
        end
        indext=[indext,ind];
        indext=indext((indext>i));   %保证是无向图，不重复
        indext=unique(indext);
        if(~isempty(indext))
            ed=ones(length(indext),2);
            ed(:,2)=i*ed(:,2);
            ed(:,1)=indext;
            edges=[edges;ed];      %构造一组无向边，存储两端节点
        end
    end

% compute affinity matrix  计算关联矩阵W  稀疏邻接
    weights = makeweights(edges,seg_vals,theta);  %theta控制边缘权重，seg_vals是每个超像素的Lab信息，计算每条无向边的权重
    W = adjacency(edges,weights,spnum);

% learn the optimal affinity matrix (eq. 3 in paper) 学习最优亲和矩阵
    dd = sum(W);                                  %按列求和输出一行向量
    D = sparse(1:spnum,1:spnum,dd); clear dd;     %度矩阵，W的列求和
    optAff =(D-alpha*W)\eye(spnum);               %alpha控制流形排序成本函数中两项的平衡，eye(spnum)返回单位方阵
    mz=diag(ones(spnum,1));                      
    mz=~mz;
    optAff=optAff.*mz;                            % (D-alpha*W)-1，对同一幅图像只需要求一次
 
%%-----------------------------stage 1--------------------------%%
% compute the saliency value for each superpixel  计算每个超像素的显著性值
% with the top boundary as the query 使用顶部边界作为查询
    Yt=zeros(spnum,1);
    bst=unique(superpixels(1,1:n));    %第一行作为顶部边界
    Yt(bst)=1;
    bsalt=optAff*Yt;                   %f=(D-alpha*W)-1*y
    bsalt=(bsalt-min(bsalt(:)))/(max(bsalt(:))-min(bsalt(:))); %归一化
    bsalt=1-bsalt;                     %求显著性

% down
    Yd=zeros(spnum,1);
    bsd=unique(superpixels(m,1:n));
    Yd(bsd)=1;
    bsald=optAff*Yd;
    bsald=(bsald-min(bsald(:)))/(max(bsald(:))-min(bsald(:)));
    bsald=1-bsald;
   
% right
    Yr=zeros(spnum,1);
    bsr=unique(superpixels(1:m,1));
    Yr(bsr)=1;
    bsalr=optAff*Yr;
    bsalr=(bsalr-min(bsalr(:)))/(max(bsalr(:))-min(bsalr(:)));
    bsalr=1-bsalr;
  
% left
    Yl=zeros(spnum,1);
    bsl=unique(superpixels(1:m,n));
    Yl(bsl)=1;
    bsall=optAff*Yl;
    bsall=(bsall-min(bsall(:)))/(max(bsall(:))-min(bsall(:)));
    bsall=1-bsall;   
   
% combine 
    bsalc=(bsalt.*bsald.*bsall.*bsalr);
    bsalc=(bsalc-min(bsalc(:)))/(max(bsalc(:))-min(bsalc(:)));
    
% assign the saliency value to each pixel     为每个具有相同标签的像素分配显著性值
     tmapstage1=zeros(m,n);
     for i=1:spnum
        tmapstage1(inds{i})=bsalc(i);
     end
     tmapstage1=(tmapstage1-min(tmapstage1(:)))/(max(tmapstage1(:))-min(tmapstage1(:)));
     
     mapstage1=zeros(w(1),w(2));                   %w(1),w(2)=m,n
     mapstage1(w(3):w(4),w(5):w(6))=tmapstage1;    %构造第一阶段显著性图
     mapstage1=uint8(mapstage1*255);  

     outname=[saldir imnames(ii).name(1:end-4) '_stage1' '.png'];
     imwrite(mapstage1,outname);

%    bsalc=seg_vals(:,1);         %各个超像素的亮度信息
%     size_b=size(bsalc,1);
%      min1=min(bsalc);
%     max1=max(bsalc);
%    for i=1:size_b
%        bsalc(i,1)=(bsalc(i,1)-min1)/(max1-min1);
%    end
     
%%----------------------stage2-------------------------%%
% binary with an adaptive threhold (i.e. mean of the saliency map) 具有自适应三重性的二进制(即显著性映射的均值)
    th=mean(bsalc);
    bsalc(bsalc<th)=0;
    bsalc(bsalc>=th)=1;%前景点，以平均值为阈值，进行二值化
    
% compute the saliency value for each superpixel 计算每个超像素的显著性值
    fsal=optAff*bsalc;    
    
% assign the saliency value to each pixel   同上
    tmapstage2=zeros(m,n);
    for i=1:spnum
        tmapstage2(inds{i})=fsal(i);    
    end
    tmapstage2=(tmapstage2-min(tmapstage2(:)))/(max(tmapstage2(:))-min(tmapstage2(:)));

mapstage2=zeros(w(1),w(2));
    mapstage2(w(3):w(4),w(5):w(6))=tmapstage2;
     gray=rgb2gray(div_log);      %转成灰度图
     mapstage2_mean=mean(mapstage2(:));
     for i=1:w(1)
         for j=1:w(2)
             if(mapstage2(i,j)>=mapstage2_mean)
                 mapstage2(i,j)=1;
             else
                 mapstage2(i,j)=0;
             end
         end
     end
     mapstage2=mapstage2.*double(gray);
    figure(2);
     imshow(uint8(3*mapstage2));
%     outname=[saldir imnames(ii).name(1:end-4) '_stage2' '.png'];   
%     imwrite(mapstage2,outname);
%     
%foreground    
    J=superpixels;
idx=find(bsalc);    %返回前景点的位置或者标签
[c,d]=size(idx);
for i=1:c
    for a=1:m       %m，n为原图像素行列
        for b=1:n
            if J(a,b)==idx(i)
                J(a,b)=NaN;
            else
                J(a,b)=J(a,b);
            end
        end
    end
end
J(~isnan(J))=0;   
J(isnan(J))=1;
outname=[foredir imnames(ii).name(1:end-4) '_foreground' '.png'];   
imwrite(J,outname);

    
% end

       
        


