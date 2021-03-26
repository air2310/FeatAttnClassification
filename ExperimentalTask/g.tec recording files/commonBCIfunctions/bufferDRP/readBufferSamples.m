function nSamples = readBufferSamples( port )

hdr = buffer('get_hdr', [], 'localhost', port);
nSamples = hdr.nsamples;