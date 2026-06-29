function Y = info_mat(P)
    P = clean_cov(P);
    Y = P \ eye(size(P));
    Y = clean_cov(Y);
end