function Pclean = clean_cov(P)
    Pclean = (P + P')/2;
    [V,D] = eig(Pclean);
    d = real(diag(D));
    d(d < 1e-10) = 1e-10;
    Pclean = V*diag(d)*V';
    Pclean = real((Pclean + Pclean')/2);
end