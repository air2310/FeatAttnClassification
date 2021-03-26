function data = readBufferData( idxSample, port )

dat = buffer('get_dat', idxSample, 'localhost', port);
data = dat.buf';