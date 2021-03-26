function map = makemap(dat)

map = NaN(3,3);
map(1,2) = dat(1);
map(2,2) = dat(2);
map(3,2) = dat(3);
map(2,1) = dat(4);
map(2,3) = dat(5);
