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

%subplot(2,3,3); plot(randWMSignalOriginal);
 title('original watermark');
authorEmbeddedWMSignalPREV = randWMSignalOriginal;
save('pqfile.mat','authorEmbeddedWMSignalPREV');

%% The DCT Transform
%  The first color in case of RGB image, grayscale information is separated from color data, so the same signal can be used for both color and black and white sets
YIQ = rgb2ntsc(inputRGBImage);
%bandRedofInput=inputRGBImage(:,:,1);
bandRedofInput=uint8(YIQ(:,:,1)*255);

%imshow(bandRedofInput);

[r,c]=size(bandRedofInput);

% Getting the DCT on Red band
imageDCT=dct2(bandRedofInput);

% Vectorizing DCT values
dctVectRedBandofInput=reshape(imageDCT,1,r*c); 
%[~,Idx]=sort(abs(dctVectRedBandofInput),'descend');%re-ordering all the absolute values
[dctVectRedBandofInput_srt,Idx]=sort(abs(dctVectRedBandofInput),'descend');
%choosing 1000 biggest values other than the DC value
Idx2=Idx(2:wmSize+1);
%%%%try work

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
   
end

 %inverse DCT to produce the watermarked asset
watermarkedImage=idct2(DCTWatermark);

imwrite(uint8(watermarkedImage), 'lena3_aa.tif', 'tif');

%Watermark Detection
watermarkedImage=imread('lena3_aa.tif', 'tif');
subplot(2,3,2); imshow(uint8(watermarkedImage)); title('Watermarked Image in Y');
% To find out if there are difference in the two images

subplot(2,3,3);
imageDifference = abs(uint8(watermarkedImage)-bandRedofInput);
imshow(imageDifference); title('Difference Image'); 

% Code to extract the watermark

imageDCT2=dct2(watermarkedImage);




%finding associated row-column order for vector values

extractedWM=zeros(1,wmSize);
%imageDCTtemp  = imageDCT;
%save('pqfile1.mat','imageDCTtemp');

for k=1:wmSize
    
    
    extractedWM(k)=[(imageDCT2(indexMAt(k,1),indexMAt(k,2))/imageDCT(indexMAt(k,1),indexMAt(k,2))-1)*10];
  
    
end

%extractedWM= round(extractedWM);
subplot(2,3,4), plot(extractedWM),title('Extracted Watermark'); %axis([-10 1010 -1 1]);


subplot(2,3,5),plot(randWMSignalOriginal);

% Plot RGB Watermarked image
%finalWMImage = inputRGBImage; 
finalWMImage = YIQ; 
%finalWMImage = watermarkedImage;
 watermarkedImage = double(watermarkedImage);
watermarkedImage = (watermarkedImage - min(watermarkedImage(:)))/ (max(watermarkedImage(:)) - min(watermarkedImage(:))) ;
finalWMImage(:,:,1) = watermarkedImage;

finalWMImage = uint8(ntsc2rgb(finalWMImage)*255);
 %subplot(2,3,5); imshow(uint8(finalWMImage*255)); title('Watermarked Image');

subplot(2,3,5); imshow(finalWMImage); title('Output Watermarked image in RGB');
%imwrite(uint8(finalWMImage), 'lena4_Final_WM.tif', 'tif');

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

subplot(2,3,6); plot(SIMs); axis([1 1000 -5 35]); title('watermark detector response'); 

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

