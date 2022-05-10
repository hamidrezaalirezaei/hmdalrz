function y=MySVRFunc(eta, H, j)
n=numel(eta);
y=0;
for i=1:n
    y=y+eta(i)*H(i,j);
end
end