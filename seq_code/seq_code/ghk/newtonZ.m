function y = newtonZ(c, m)
    tol = 10e-10;
    dist = 10;
    while dist > tol 
        mnew = (normpdf(m) - c)/(1 - normcdf(m));
        dist = abs(mnew - m);
        m = mnew;
    end
    y = m;
end
    
    