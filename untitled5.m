% dyn=load('./N=6000_omg=3.32/data/real3d-dyna.txt');
ff='amp=0.05_freq=2.236';
dyn=load([ff,'/real3d-dyna.txt']);

t=dyn(:,1);
nt=length(t);
dt = t(2)-t(1);

zt = dyn(:,10);
deltaxt = dyn(:,7);
deltazt = dyn(:,11);
if ~mod(nt,2)
    w = 2*pi/(nt*dt)*(-nt/2:nt/2-1);
else
    w = 2*pi/(nt*dt)*(-(nt-1)/2:(nt-1)/2);
end


zw = fftshift(fft(zt-mean(zt)));
deltaxw = fftshift(fft(deltaxt - mean(deltaxt)));
deltazw = fftshift(fft(deltazt - mean(deltazt)));

figure
subplot(321)
plot(t,zt)
ylabel("z(t)")
subplot(323)
plot(t,deltaxt)
ylabel("\Delta x(t)")
subplot(325)
plot(t,deltazt)
ylabel("\Delta z(t)")
xlabel("\omega_0 t")

subplot(322)
plot(w,abs(zw))
ylabel("z(f)")
xlim([-pi,pi])
subplot(324)
plot(w,abs(deltaxw))
ylabel("\Delta x(f)")
xlim([-pi,pi])
subplot(326)
plot(w,abs(deltazw))
ylabel("\Delta z(f)")
xlim([-pi,pi])
xlabel('f')