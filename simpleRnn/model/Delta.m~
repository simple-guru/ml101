function [delta2x delta2h delta3] = Delta(a1,z2,a2,Why,z3,a3,yk)
delta3 = a3 .- yk;
fprintf('\nSize of delta3\n');
size(delta3)	

delta2x =( (Why' * delta3') .* (a2' .* (1 - a2')) );
fprintf('\nSize of delta2x \n');
size(delta2x)
%pause

delta2x = delta2x(2:end);
delta2h = delta2x;


fprintf('\nSize of a1\n');
size(a1)
fprintf('\nSize of a2\n');
size(a2)
fprintf('\nSize of a3\n');
size(a3)

fprintf('\nSize of yk\n');
size(yk)

fprintf('\nSize of z2\n');
size(z2)

fprintf('\nSize of z3\n');
size(z3)

fprintf('\nSize of Why\n');
size(Why)


fprintf('\nSize of delta2x delta2h\n');
size(delta2x)
size(delta2h)

%pause

end
