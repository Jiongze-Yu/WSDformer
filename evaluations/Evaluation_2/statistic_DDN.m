clear all;
ts =0;
tp =0;
for i=1:100                          % the number of testing samples DDN-Data
   x_true=im2double(imread(strcat('..\..\datasets\DDN-Data\test\gt\',sprintf('%d.jpg',900+i))));  % groundtruth 
   x_true = rgb2ycbcr(x_true);
   x_true = x_true(:,:,1); 
   for j=1:14
       x = im2double(imread(strcat('..\..\results\WSDformer_DDN-Data\',sprintf('%d_%d.jpg',900+i,j))));     %reconstructed image
       x = rgb2ycbcr(x);
       x = x(:,:,1);
       m = size(x,1);
       n = size(x,2);
       x_true = x_true(1:m,1:n,:);
       tp= tp+ psnr(x,x_true);
       ts= ts+ssim(x*255,x_true*255);
   end
end
fprintf('psnr=%6.4f, ssim=%6.4f\n',tp/1400,ts/1400)                          % the number of testing samples DDN-Data



