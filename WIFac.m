function [Results_WIFac] = WIFac(Data_Real,Data_Virtual)

size_Real = size(Data_Real);
size_Virtual = size(Data_Virtual);

n_Real_x = size_Real(1,1);
n_Real_y = size_Real(1,2)/2;
n_Virtual_x = size_Virtual(1,1);
n_Virtual_y = size_Virtual(1,2)/2;

for i = 1:n_Real_y
    x_Real(:,i) = Data_Real(:,2*i-1);
    y_Real(:,i) = Data_Real(:,2*i);
end

for i = 1:n_Virtual_y
    x_Virtual(:,i) = Data_Virtual(:,2*i-1);
    y_Virtual(:,i) = Data_Virtual(:,2*i);
end

f_Real = round(y_Real,8);
g_Virtual = round(y_Virtual,8);

f_nsquare_Real = power(f_Real,2);
g_nsquare_Virtual = power(g_Virtual,2);
fg_product = f_Real(:,n_Real_y).*g_Virtual;

for j = 1:n_Virtual_y
    for i = 1:n_Virtual_x
        if f_nsquare_Real(i,n_Real_y) == 0
        else if g_nsquare_Virtual(i,j) == 0
                numerator_Real_Virtual(i,j) = 0;
            else
                numerator_Real_Virtual(i,j) = max(f_nsquare_Real(i,n_Real_y),g_nsquare_Virtual(i,j)).*((1-(max(0,fg_product(i,j))/max(f_nsquare_Real(i,n_Real_y),g_nsquare_Virtual(i,j)))).^2);   % 분자항 계산
                denominator_Real_Virtual(i,j) = max(f_nsquare_Real(i,n_Real_y),g_nsquare_Virtual(i,j));   % 분모항 계산
            end
        end
    end
end

for i = 1:n_Virtual_y
    DoC_Real_Virtual(i) = 1-nthroot((sum(numerator_Real_Virtual(:,i))/sum(denominator_Real_Virtual(:,i))),2);   % WIFac 계산
end
    Results_WIFac = transpose(DoC_Real_Virtual);    
end