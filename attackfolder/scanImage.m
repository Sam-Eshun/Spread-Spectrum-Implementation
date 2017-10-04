clc;
close all;
clear;


inputRGBImage=imread('laa.jpg');
%filter = fspecial('gaussian', [5 5], 0.1);
%filter = fspecial('motion', 4, 0);
%inputRGBImage = imfilter(inputRGBImage, filter);
%imshow(inputRGBImage);
imshow('laa.jpg');
ll = imread('lena3.tif');

dctOrg = dct2(ll(:,:,1));
%inputRGBImage = imcrop(inputRGBImage,[40 100 435 310]);
%inputRGBImage = imresize(inputRGBImage, 1.5);
subplot(2,3,1); imshow(inputRGBImage); title('Input Image');

TypeOfDistribution = 'v5uniform';

%alpha = 0.1;
%alpha = 100
noOfRandomMarks = 1000;


%% Gaussian Watermark generation
% The watermark size
wmSize=1000; %watermark size
% Generating Gaussian random vector to use as watermark(W)
randWMSignalOriginal=randn(1,wmSize);
subplot(2,3,3); plot(randWMSignalOriginal);
authorEmbeddedWMSignalPREV = randWMSignalOriginal;
save('pqfile.mat','authorEmbeddedWMSignalPREV');
%% The DCT Transform
%  The first color in case of RGB image, grayscale information is separated from color data, so the same signal can be used for both color and black and white sets
bandRedofInput=inputRGBImage(:,:,1);
[r,c]=size(bandRedofInput);

% Getting the DCT on Red band
imageDCT=dct2(bandRedofInput);

dctVectRedBandofInput=reshape(imageDCT,1,r*c); % Vectorizing DCT values
[~,Idx]=sort(abs(dctVectRedBandofInput),'descend');%re-ordering all the absolute values
%[dctVectRedBandofInput_srt,Idx]=sort(abs(dctVectRedBandofInput),'descend');
Idx2=Idx(2:wmSize+1);%choosing 1000 biggest values other than the DC value

%finding associated row-column order for vector values
indexMAt=zeros(wmSize,2);

for k=1:wmSize
x=floor(Idx2(k)/r)+1;%associated culomn in the image
y=mod(Idx2(k),r);%associated row in the image
indexMAt(k,1)=y; % Row Index
indexMAt(k,2)=x; % Column Index
end
DCTWatermark=imageDCT;
%insert the WM signal into the DCT values

% for k=1:wmSize
%     %DCTWatermark(indexMAt(k,1),indexMAt(k,2))=DCTWatermark(indexMAt(k,1),indexMAt(k,2))+alpha*DCTWatermark(indexMAt(k,1),indexMAt(k,2)).*randWMSignalOriginal(k);
%     %  D_w(IND(k,1),IND(k,2))=D_w(IND(k,1),IND(k,2))+.1*D_w(IND(k,1),IND(k,2)).*W(k);
% end

%figure,imshow(D_w);
watermarkedImage=idct2(DCTWatermark); %inverse DCT to produce the watermarked asset

%imwrite(uint8(watermarkedImage), 'lena3_aa.tif', 'tif');

%Watermark Detection
%watermarkedImage=imread('lena3_aa.tif', 'tif');
subplot(2,3,2); imshow(uint8(watermarkedImage)); title('Watermarked Image');
% To find out if there are difference in the two images

subplot(2,3,3);
%imageDifference = abs(uint8(watermarkedImage)-bandRedofInput);
%imshow(imageDifference); title('Difference Image'); 

% Code to extract the watermark
%I3=watermarkedImage(:,:,1); %dct2(watermarkedImage
%extractedWM= dct2(I3);
%imageDCT1=dct2(watermarkedImage);



%extractedWM =[];
extractedWM=zeros(1,wmSize);
imageDCTtemp  = imageDCT;
save('pqfile1.mat','imageDCTtemp');
sdf =DCTWatermark(1,1);

for k=1:wmSize
    extractedWM(k)=[(DCTWatermark(indexMAt(k,1),indexMAt(k,2))/dctOrg(indexMAt(k,1),indexMAt(k,2))-1)*10];
  %W2(k)=[(D_w(IND(k,1),IND(k,2))/D(IND(k,1),IND(k,2))-1)*10];
    
end

%extractedWM= round(extractedWM);

subplot(2,3,4);
%title('Extracted Watermark'); 
plot(extractedWM);  title('extracted watermark from scanned image');
axis([0 1000 -5 10]);


%subplot(2,3,5),plot(randWMSignalOriginal);

% Plot RGB Watermarked image
finalWMImage = inputRGBImage; 
%finalWMImage = watermarkedImage;
finalWMImage(:,:,1) = watermarkedImage;
subplot(2,3,5); imshow(finalWMImage); title('Watermarked Image');
imwrite(uint8(finalWMImage), 'lena4_Final_WM.tif', 'tif');

psnr = myPsnr(inputRGBImage,ll);


%% Compute the SIM
%SIM =zeros(1,noOfRandomMarks);
SIM = abs(randWMSignalOriginal * extractedWM' / sqrt(extractedWM * extractedWM'));

%% Random signal attack and Smilarity check
randomMarks=round(rand(noOfRandomMarks,wmSize));
randomMarks(200,:)=randWMSignalOriginal; %50th row is the original watermark
SIMs=zeros(1,noOfRandomMarks);

for k = 1:1:noOfRandomMarks
   % if(k==200)
     randomMark =randomMarks(k,:);
     %SIMs(k) = abs(randomMarks(k,:)*extractedWM'/sqrt(extractedWM*extractedWM'));
     SIMs(k) = abs(randWMSignalOriginal*randomMarks(k,:)'/sqrt(randomMarks(k,:)*randomMarks(k,:)'));
%     

 %end
end

subplot(2,3,6); plot(SIMs); axis([1 1000 -5 35]); title('500 = original watermark'); 

%  
  %plot(SIMs(200));
%  %axis([1 1000 -5 35]);




% Set threshold

SIMs = [SIMs(1:199) SIMs(201:end)];
threshold = max(SIMs)*0.8;

if (SIM >threshold)
    disp('The Watermark is found');
else
    disp('The Watermark is lost');         
end
%% 














%% the output in the same colour

