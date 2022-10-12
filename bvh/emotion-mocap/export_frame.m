clear;
bvhdir = 'G:\BVH\F01';
subdir = dir(bvhdir);
frames = {};
totalname={};
for diri = 1:length(subdir)
    
    if( isequal( subdir( diri ).name, '.' )||...
            isequal( subdir( diri ).name, '..')||...
            ~subdir( diri ).isdir)               % 如果不是目录则跳过
        continue;
    end
    subdirpath = fullfile( bvhdir, subdir( diri ).name, '*.bvh' );
    files= dir(subdirpath);
    cd(fullfile( bvhdir, subdir( diri ).name))
    for i = 1:length(files)
        filename = files(i).name;
        fid = fopen(filename);
        txtdata = fscanf(fid,'%s');
        idx = strfind(txtdata,'Frames:');
        temp=txtdata(idx+7:idx+10);
        fclose(fid);
        if isequal(temp(end),'F')
            temp = temp(1:3);
        end
        frames = [frames;{temp}];
        totalname = [totalname;{filename}];
    end
end

data= [totalname,frames];
cd(bvhdir)
xlswrite('frames.xlsx',data);