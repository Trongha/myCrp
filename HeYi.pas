Program HeYi(output);

var A, C : array[1..1000] of integer;
var n, i, dem, daiNhat, start : integer;
begin
    write('Số Phần Tử: ');
    readln(n);
    
    writeln('Nhap mang A: ');
    for i:=1 to n do
        begin
            write('A',i,' = ');
            readln(C[i]);
        end;
        
    writeln('Nhap mang C: ');
    for i:=1 to n do
        begin
            write('C',i,' = ');
            readln(C[i]);
        end;

    daiNhat := 0;
    dem := 0;
    
    for i:=1 to n do
        if (A[i] < C[i]) then
            begin
                start := i;
                while ((i+1 <= n) and (A[i] < C[i])) do
                    i:= i+1;
                dem := i-start+1;
                if (dem > daiNhat) then
                    daiNhat := dem;
            end;
    
    write('dem = ', dem);
end.