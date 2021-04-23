%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ShowDRIONS_DB
% 
% This is a MATLAB program to show the 110 eye fundus images and optic
% nerve contours (expert 1 and expert2) wich contains DRIONS_DB. Each 
% image is displayed one after another by pressing any key. You will have
% to update the variables 'path_ExpertContours' and 'path_FO_Images' which
% have to contain the paths where you unzipped the folders 'experts_anotation' and
% 'images', respectively.
%
% Date: 2009 08 19
% Author: Enrique Carmona
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Images and Contour Paths
path_ExpertContours='C:\Ejcasu\Matlab\work\MutuaMad\DRIONS-DB\experts_anotation\';
path_FO_Images='C:\Ejcasu\Matlab\work\MutuaMad\DRIONS-DB\images\';

for i=1:110
    %Build complete path for each image and contour file
    if i<10
        path_Expert1Contour=[path_ExpertContours,'anotExpert1_00',num2str(i),'.txt'];
        path_Expert2Contour=[path_ExpertContours,'anotExpert2_00',num2str(i),'.txt'];
        path_FO_Image=[path_FO_Images,'image_00',num2str(i),'.jpg'];
    else if i<100
            path_Expert1Contour=[path_ExpertContours,'anotExpert1_0',num2str(i),'.txt'];
            path_Expert2Contour=[path_ExpertContours,'anotExpert2_0',num2str(i),'.txt'];
            path_FO_Image=[path_FO_Images,'image_0',num2str(i),'.jpg'];
        else
            path_Expert1Contour=[path_ExpertContours,'anotExpert1_',num2str(i),'.txt'];
            path_Expert2Contour=[path_ExpertContours,'anotExpert2_',num2str(i),'.txt'];
            path_FO_Image=[path_FO_Images,'image_',num2str(i),'.jpg'];
        end
    end
       
    
    %Load contours files (from expert1 and expert2)
    Exp1Contour=load(path_Expert1Contour);
    Exp2Contour=load(path_Expert2Contour);
    %Load eye fundus image
    img_FO=imread(path_FO_Image);

    %Show eye fundus image and contours
    imshow(img_FO)
    hold on
    plot(Exp1Contour(:,1),Exp1Contour(:,2),'g.-');
    plot(Exp2Contour(:,1),Exp2Contour(:,2),'b.-');
    xlabel(['Image ',num2str(i)]);
    
    %Press any key to continue
    pause
end