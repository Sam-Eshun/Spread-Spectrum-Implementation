clc;
close all;
clear;

inputRGBImage=imread('lena4_Final_WM.tif');
originalimage=imread('lena3.tif');
subplot(2,3,1); imshow(inputRGBImage); title('Input Image'); 
TypeOfDistribution = 'v5uniform';
alpha = 0.1; noOfRandomMarks = 1000;
load('pqfile.mat');
%% Gaussian Watermark generation
% The watermark size
wmSize=1000; %watermark sizerr
% Generating Gaussian random vector to use as watermark(W)
noOfWMs = 5;
authorEmbeddedWMSignal=randn(noOfWMs,wmSize);

%% The DCT Transform
%  The first color in case of RGB image, grayscale information is separated from color data, so the same signal can be used for both color and black and white sets
bandRedofInput=inputRGBImage(:,:,1);
[r,c]=size(bandRedofInput);

individualWMImage = {};

for waterMarkCount = 1:1:size(authorEmbeddedWMSignal,1)
    
    % Getting the DCT on Red band
    imageDCT=dct2(double(bandRedofInput));
    
    dctVectRedBandofInput=reshape(imageDCT,1,r*c); % Vectorizing DCT values
    [~,Idx]=sort(abs(dctVectRedBandofInput),'descend');%re-ordering all the absolute values
     %choosing 1000 biggest values other than the DC value
    Idx2=Idx(2:wmSize+1);

    %finding associated row-column order for vector values
    indexMAt=zeros(wmSize,2);
    for k=1:wmSize
    x=floor(Idx2(k)/r)+1;%associated culomn in the image
    y=mod(Idx2(k),r);%associated row in the image
    indexMAt(k,1)=y; % Row Index
    indexMAt(k,2)=x; % Column Index
    end
imageDCTBeforeWm= imageDCT;
    %insert the WM signal into the DCT values
    for k=1:wmSize
        imageDCT(indexMAt(k,1),indexMAt(k,2))=imageDCT(indexMAt(k,1),indexMAt(k,2))+alpha*imageDCT(indexMAt(k,1),indexMAt(k,2)).*authorEmbeddedWMSignal(waterMarkCount,k);
    end
    
    %inverse DCT to produce the watermarked asset
    
    watermarkedImage=idct2(imageDCT); 
    finalWMImage = inputRGBImage;
    finalWMImage(:,:,1) = watermarkedImage;
    %subplot(2,3,waterMarkCount); imshow(finalWMImage);
    individualWMImage{waterMarkCount} = finalWMImage;
    %imwrite(uint8(finalWMImage), strcat('lena4_Final_WM',num2str(waterMarkCount),'.tif'), 'tif');
end

% Averging of 5 watermarked images
avergagedImage = double(zeros(r,c,3));
for i = 1:1:size(individualWMImage,2)
    imgs = double(individualWMImage{i});
    avergagedImage = imadd(avergagedImage, imgs);
end
avergagedImage = avergagedImage./size(individualWMImage,2);
subplot(2,3,2); imshow(uint8(avergagedImage)); title('Average WMarked Image'); 

watermarkedImage=avergagedImage(:,:,1);
subplot(2,3,2); imshow(uint8(watermarkedImage)); title('Watermarked Image');

% To find out if there are difference in the two images
subplot(2,3,3);
imageDifference = abs(uint8(watermarkedImage)-bandRedofInput);
imshow(imageDifference); title('Difference Image'); 


psnr = myPsnr(originalimage,avergagedImage);




load('pqfile.mat');
load('pqfile1.mat');
wmFromPreviousCode= authorEmbeddedWMSignalPREV;
OriginalImageDCTFRomPrevCode =imageDCTtemp;


% Code to extract the watermark
I3=dct2(watermarkedImage(:,:,1));
%W2=dct2(I3);
extractedWM=zeros(1,wmSize);
for k=1:wmSize
   
    extractedWM(k)=[(I3(indexMAt(k,1),indexMAt(k,2))/double(imageDCTBeforeWm(indexMAt(k,1),indexMAt(k,2)))-1)*10];
end
subplot(2,3,4), plot(extractedWM),title('Extracted Watermark'); %axis([-10 1010 -1 1]);

% Plot RGB Watermarked image
subplot(2,3,5); imshow(finalWMImage); title('Watermarked Image(RGB)');

%% Computing the similarities to check if ...
SIM =abs(extractedWM*authorEmbeddedWMSignal'/sqrt(authorEmbeddedWMSignal*authorEmbeddedWMSignal'));

%% Random signal attack and Smilarity check
randomMarks=round(rand(noOfRandomMarks,wmSize));
for wCount = 1:1:size(authorEmbeddedWMSignal,1)
    randomMarks(wCount*150,:) = authorEmbeddedWMSignal(wCount,:); %200th row is the original watermark
    randomMarks(wCount*500,:) =  wmFromPreviousCode;
    SIMs=zeros(1,noOfRandomMarks);
    for k = 1:1:noOfRandomMarks
        randomMark =randomMarks(k,:);
        SIMs(k) = (abs(extractedWM)*randomMarks(k,:)')/(sqrt(randomMarks(k,:)*randomMarks(k,:)'));
    end
    SIMs = -SIMs + max(SIMs);
    subplot(2,3,6); plot(SIMs(:)); hold on; axis([1 1000 0 10]); title('200 = original watermark');
end
hold off;

% Set threshold
threshold = max(SIM)*0.8;
if (SIMs(150) > threshold)
    if(SIMs(300) > threshold)
        if(SIMs(450) > threshold)
            if(SIMs(600) > threshold)
                disp('The Watermark is found');
            else
                disp('The Watermark is lost');   
            end
        else
            disp('The Watermark is lost');   
        end
    else
        disp('The Watermark is lost');   
    end
else
    disp('The Watermark is lost');         
end



















