function m_output = contractionZ(m, c)
    tol = 10e-10;
    dist = 10;
    while dist > tol 
        mnew = (m)*normcdf(m) + normpdf(m) - c;
        dist = abs(mnew - m);
        m = mnew;
    end
    m_output = m;
end