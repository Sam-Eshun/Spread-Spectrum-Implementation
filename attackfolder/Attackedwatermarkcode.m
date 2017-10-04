clc;
close all;
clear;

inputRGBImage=imread('lena3.tif');

subplot(2,3,1); imshow(inputRGBImage); title('Input Image'); 
TypeOfDistribution = 'v5uniform';

alpha = 0.1;
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

YIQ = rgb2ntsc(inputRGBImage);
%bandRedofInput=inputRGBImage(:,:,1);
bandRedofInput=uint8(YIQ(:,:,1)*255);
%bandRedofInput=inputRGBImage(:,:,1);
[r,c]=size(bandRedofInput);

% Getting the DCT on Red band
imageDCT=dct2(bandRedofInput);

dctVectRedBandofInput=reshape(imageDCT,1,r*c); % Vectorizing DCT values
%[~,Idx]=sort(abs(dctVectRedBandofInput),'descend');%re-ordering all the absolute values
[dctVectRedBandofInput_srt,Idx]=sort(abs(dctVectRedBandofInput),'descend');
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
for k=1:wmSize
    DCTWatermark(indexMAt(k,1),indexMAt(k,2))=DCTWatermark(indexMAt(k,1),indexMAt(k,2))+alpha*DCTWatermark(indexMAt(k,1),indexMAt(k,2)).*randWMSignalOriginal(k);
    %  D_w(IND(k,1),IND(k,2))=D_w(IND(k,1),IND(k,2))+.1*D_w(IND(k,1),IND(k,2)).*W(k);
end

%figure,imshow(D_w);
watermarkedImage=idct2(DCTWatermark); %inverse DCT to produce the watermarked asset

imwrite(uint8(watermarkedImage), 'lenatoattack.tif', 'tif');

%% Attacking the watermark
%watermarkedImage=imrotate(watermarkedImage,90);
%imshow(watermarkedImage/256);
%Noise attack

watermarkedImage2=imread('lenatoattack.tif', 'tif');
%Lena = imread('lena.tif');
%watermarkedImage3 = imnoise(watermarkedImage2,'salt & pepper',0.02);
%imshow(watermarkedImage3);
watermarkedImage3 = imnoise(watermarkedImage2,'gaussian',0.02);
%watermarkedImage3 = imnoise(watermarkedImage2,'gaussian',0.04);
%watermarkedImage3 = imnoise(watermarkedImage2,'gaussian',0.06);
%watermarkedImage3 = imnoise(watermarkedImage2,'gaussian',0.08);
%watermarkedImage3 = imnoise(watermarkedImage2,'gaussian',0.10);
%imwrite(uint8(watermarkedImage3), 'lena6.tif', 'tif');
%imshow(watermarkedImage3);
%This filter can not remove the mark
 %filter = fspecial('average'); 

%These filters can remove the watermark if it's not robust
%filter = fspecial('gaussian', [5 5], 0.50);
filter = fspecial('gaussian', [5 5], 1.5);
%filter = fspecial('gaussian', [5 5], 1.50);
%filter = fspecial('gaussian', [5 5], 2.00);
%filter = fspecial('gaussian', [5 5], 2.50);
%b=conv2(watermarkedImage3,filter,'same');
%figure,imshow(b/256);
%filter = fspecial('gaussian', [5 5], 0.5);
%filter = fspecial('motion', 4, 0);
watermarkedImage3 = imfilter(watermarkedImage3, filter);





%imshow(watermarkedImage3);
%watermarkedImage1=watermarkedImage3;
% Image = imread('lena6.tif');
 %figure, imshow('lena6.tif');

%crop Image
Noiseimage=watermarkedImage3;
%Noiseimage=imread('lena6.tif');
% 
  I2 = imcrop(Noiseimage,[180 179 179 179]);
%  
 YIQ = rgb2ntsc(inputRGBImage);
%  
 %bandRedofInput=inputRGBImage(:,:,1);
 bandRedofInput=YIQ(:,:,1);
  
  I2org = imcrop(bandRedofInput,[180 179 179 179]);
  
  subplot(2,3,2); 
  Noiseimage(180:180+179,179:179+179,1)= uint8(I2org(:,:,1)*255);
 %figure, imshow(Noiseimage)
 %Noiseimage(180:180+179,179:179+179,2)= I2org(:,:,2);
 %Noiseimage(180:180+179,179:179+179,3)= I2org(:,:,3);
 %[rows, columns, numberOfColorChannels] = size(inputRGBImage);

   %resize to match original image ,
%I2 = imresize(I2, [rows, columns]);
I2= Noiseimage;
 
 
 
 
% imshow(I2)
% subplot(1,2,1)
% imshow(Noiseimage)
% title('Original Image')
% subplot(1,2,2)
% imshow(I2)
% title('Cropped Image')


%%
%Watermark Detection
%watermarkedImage=imrotate(watermarkedImage,90);
watermarkedImage=I2;




%subplot(2,3,2); imshow(uint8(I2)); title('Watermarked Image');
subplot(2,3,2); imshow(uint8(watermarkedImage)); title('Watermarked Image');
% To find out if there are difference in the two images

%subplot(2,3,3);
%imageDifference = abs(uint8(watermarkedImage)-bandRedofInput);
%imshow(imageDifference); title('Difference Image'); 

% Code to extract the watermark
%I3=watermarkedImage(:,:,1);   % dct2(watermarkedImage)
%extractedWM= dct2(I3);

%figure,imshow(watermarkedImage);

dctNoiseImage = dct2(watermarkedImage);

extractedWM=zeros(1,wmSize);
imageDCTtemp  = imageDCT;
save('pqfile1.mat','imageDCTtemp');
for k=1:wmSize
    extractedWM(k)=[(dctNoiseImage(indexMAt(k,1),indexMAt(k,2))/imageDCT(indexMAt(k,1),indexMAt(k,2))-1)*10];
  %W2(k)=[(D_w(IND(k,1),IND(k,2))/D(IND(k,1),IND(k,2))-1)*10];
    
end

%extractedWM= round(extractedWM);
subplot(2,3,4), plot(extractedWM),title('Extracted Watermark'); %axis([-5 1000 -2 2]);


%subplot(2,3,5),plot(randWMSignalOriginal);

% Plot RGB Watermarked image
 %finalWMImage = inputRGBImage; 
 finalWMImage = YIQ; 
 watermarkedImage = double(watermarkedImage);
 watermarkedImage = (watermarkedImage - min(watermarkedImage(:)))/(max(watermarkedImage(:)) - min(watermarkedImage(:))) ;
% %finalWMImage = watermarkedImage;
 finalWMImage(:,:,1) = watermarkedImage;
 finalWMImage = ntsc2rgb(finalWMImage);
 subplot(2,3,5); imshow(uint8(finalWMImage*255)); title('Watermarked Image');
% imwrite(uint8(finalWMImage), 'lena4_Final_WM.tif', 'tif');

psnr = myPsnr(inputRGBImage,finalWMImage);


%% Compute the SIM
SIM = abs(randWMSignalOriginal * extractedWM' / sqrt(extractedWM * extractedWM'));

%% Random signal attack and Smilarity check
randomMarks=round(rand(noOfRandomMarks,wmSize));
randomMarks(200,:)=randWMSignalOriginal; %50th row is the original watermark
SIMs=zeros(1,noOfRandomMarks);

for k = 1:noOfRandomMarks
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

threshold = max(SIMs)*1.1;

if (SIM >threshold)
    disp('The Watermark is found');
else
    disp('The Watermark is lost');         
end
%% 




%% the output in the same colour

