program part2cube
  !--------------------------------------------------------------------------
  ! Build a 3D density cube from MINI-RAMSES particle outputs.
  ! Supports NGP, CIC, TSC, PCS deposition (tensor-product B-splines).
  ! Output format identical to ramses part2cube.f90
  !--------------------------------------------------------------------------
  implicit none

  integer,parameter::flen=200
  integer,parameter::DEP_NGP=0, DEP_CIC=1, DEP_TSC=2, DEP_PCS=3

  integer::npart,nfile
  integer::nx_sample=128,ny_sample=128,nz_sample=128,nx,ny,nz
  integer::i,icpu
  integer::npart_file, ndim_file, npart_actual
  integer(kind=8)::ipos
  integer::dep_scheme=DEP_CIC

  real(kind=8)::xmin=-1,xmax=-1,ymin=-1,ymax=-1,zmin=-1,zmax=-1
  real(kind=8)::xxmin,xxmax,yymin,yymax,zzmin,zzmax
  real(kind=8)::dx,dy,dz,xx,yy,zz,mtot
  real(kind=8)::ddx,ddy,ddz

  character(len=flen)::nomfich,repository,prefix='part',outfich
  character(len=flen)::file_part
  character(len=8)::depname
  character(len=5)::ncharcpu

  logical::ok_part,periodic=.false.,check_ramses_exist

  real(kind=8),dimension(:,:,:),allocatable::cube
  real(kind=4),dimension(:,:,:),allocatable::toto
  real(kind=4),dimension(:,:),allocatable::x
  real(kind=4),dimension(:),allocatable::m

  type params
     integer::ndim
     integer::ncpu
     integer::nvar
     integer::levelmin
     integer::nlevelmax
     real(kind=8)::boxlen
     real(kind=8)::t
     real(kind=8)::texp
     real(kind=8)::aexp
     real(kind=8)::gamma
     real(kind=8)::unit_d
     real(kind=8)::unit_l
     real(kind=8)::unit_t
  end type params
  type(params)::p

  !------------------------------
  ! Read parameters and headers
  !------------------------------
  call read_params

  if(.NOT. check_ramses_exist(repository,prefix))then
     write(*,*)'Repository '//trim(repository)//' incomplete.'
     stop
  endif

  call read_ramses_params
  call read_info
  call read_part_header

  write(*,*) 'time =', real(p%t,kind=4)
  write(*,*) 'npart=', npart
  write(*,*) 'nfile=', nfile

  !-----------------
  ! Set up geometry
  !-----------------
  if(xmin<0)xmin=0
  if(xmax<0)xmax=p%boxlen
  if(ymin<0)ymin=0
  if(ymax<0)ymax=p%boxlen
  if(zmin<0)zmin=0
  if(zmax<0)zmax=p%boxlen

  xxmin=xmin ; xxmax=xmax
  yymin=ymin ; yymax=ymax
  zzmin=zmin ; zzmax=zmax

  nx=nx_sample
  ny=ny_sample
  nz=nz_sample

  dx=(xxmax-xxmin)/dble(nx)
  dy=(yymax-yymin)/dble(ny)
  dz=(zzmax-zzmin)/dble(nz)

  ! Allocate cube (accumulate in double for accuracy)
  npart_actual=0
  mtot=0.0d0
  allocate(cube(0:nx,0:ny,0:nz))
  cube=0.0d0

  !----------------------------
  ! Read and deposit particles
  !----------------------------
  do icpu=1,nfile
     call title(icpu,ncharcpu)
     file_part=trim(repository)//'/'//trim(prefix)//'.'//trim(ncharcpu)
     open(unit=10,file=file_part,access='stream',action='read',form='unformatted')
     ipos=1
     read(10,POS=ipos)ndim_file
     ipos=5
     read(10,POS=ipos)npart_file
     allocate(x(1:npart_file,1:ndim_file))
     allocate(m(1:npart_file))
     ipos=9
     read(10,POS=ipos)x
     ipos=9+4*int(npart_file,kind=8)*2*int(ndim_file,kind=8)
     read(10,POS=ipos)m

     do i=1,npart_file
        ok_part=(x(i,1)>=xmin.and.x(i,1)<=xmax.and.&
     &           x(i,2)>=ymin.and.x(i,2)<=ymax.and.&
     &           x(i,3)>=zmin.and.x(i,3)<=zmax)
        if(.not.ok_part)cycle
        npart_actual=npart_actual+1

        ddx=(dble(x(i,1))-xxmin)/dx
        ddy=(dble(x(i,2))-yymin)/dy
        ddz=(dble(x(i,3))-zzmin)/dz

        select case(dep_scheme)
        case(DEP_NGP)
          call deposit_ngp3d(cube,nx,ny,nz,ddx,ddy,ddz,dble(m(i)),periodic)
          mtot=mtot+dble(m(i))
        case(DEP_CIC)
          call deposit_cic3d(cube,nx,ny,nz,ddx,ddy,ddz,dble(m(i)),periodic)
          mtot=mtot+dble(m(i))
        case(DEP_TSC)
          call deposit_tsc3d(cube,nx,ny,nz,ddx,ddy,ddz,dble(m(i)),periodic)
          mtot=mtot+dble(m(i))
        case(DEP_PCS)
          call deposit_pcs3d(cube,nx,ny,nz,ddx,ddy,ddz,dble(m(i)),periodic)
          mtot=mtot+dble(m(i))
        end select
     end do

     deallocate(x,m)
     close(10)
  end do

  write(*,*)'Data read and deposited.'
  write(*,*)'Total number of part=',npart_actual
  write(*,*)'Total deposited mass=',mtot

  !-----------------
  ! Write output
  !-----------------
  outfich=trim(prefix)//'.cube'
  write(*,*)'Writing data to ', trim(outfich)
  open(unit=20,file=outfich,form='unformatted')
  if(periodic)then
     write(20)nx,ny,nz
     allocate(toto(0:nx-1,0:ny-1,0:nz-1))
     toto=real(cube(0:nx-1,0:ny-1,0:nz-1),kind=4)
     write(20)toto
  else
     write(20)nx+1,ny+1,nz+1
     allocate(toto(0:nx,0:ny,0:nz))
     toto=real(cube(0:nx,0:ny,0:nz),kind=4)
     write(20)toto
  endif
  close(20)

contains

  subroutine read_params
    implicit none
    integer::ii,n
    character(len=4)::opt
    character(len=128)::arg

    n = command_argument_count()
    if (n < 4) then
       print *, 'usage: part2cube -inp input_dir -pre prefix'
       print *, '                 [-dep NGP|CIC|TSC|PCS]'
       print *, '                 [-xmi xmin] [-xma xmax]'
       print *, '                 [-ymi ymin] [-yma ymax]'
       print *, '                 [-zmi zmin] [-zma zmax]'
       print *, '                 [-nx nx] [-ny ny] [-nz nz]'
       print *, '                 [-per periodic]'
       print *, 'ex: part2cube -inp output_00001 -pre part -dep TSC'
       stop
    end if

    do ii=1,n,2
       call get_command_argument(ii,opt)
       if (ii == n) then
          print '("option ",a2," has no argument")', opt
          stop 2
       end if
       call get_command_argument(ii+1,arg)
       select case (opt)
       case ('-inp')
          repository=trim(arg)
       case ('-pre')
          prefix=trim(arg)
       case ('-dep')
          depname=trim(arg)
          if(depname=='NGP' .or. depname=='ngp' .or. depname=='0')then
             dep_scheme=DEP_NGP
          elseif(depname=='CIC' .or. depname=='cic' .or. depname=='1')then
             dep_scheme=DEP_CIC
          elseif(depname=='TSC' .or. depname=='tsc' .or. depname=='2')then
             dep_scheme=DEP_TSC
          elseif(depname=='PCS' .or. depname=='pcs' .or. depname=='3')then
             dep_scheme=DEP_PCS
          else
             write(*,*)'Unknown deposition: ',trim(depname),', defaulting to CIC'
             dep_scheme=DEP_CIC
          endif
       case ('-xmi')
          read (arg,*) xmin
       case ('-xma')
          read (arg,*) xmax
       case ('-ymi')
          read (arg,*) ymin
       case ('-yma')
          read (arg,*) ymax
       case ('-zmi')
          read (arg,*) zmin
       case ('-zma')
          read (arg,*) zmax
       case ('-nx')
          read (arg,*) nx_sample
       case ('-ny')
          read (arg,*) ny_sample
       case ('-nz')
          read (arg,*) nz_sample
       case ('-per')
          read (arg,*) periodic
       case default
          print '("unknown option ",a2," ignored")', opt
       end select
    end do
  end subroutine read_params

  subroutine read_ramses_params
    ! Read params.bin to get ndim, nfile/ncpu, boxlen, times, gamma
    character(len=128)::fn
    integer::ilun,noutput,skip
    integer::nfile1,ncpu1
    fn=trim(repository)//'/params.bin'
    ilun=11
    open(unit=ilun,file=fn,access='stream',action='read',form='unformatted')
    read(ilun,POS=1)nfile1
    read(ilun,POS=5)ncpu1
    read(ilun,POS=9)p%ndim
    read(ilun,POS=13)p%levelmin
    read(ilun,POS=17)p%nlevelmax
    read(ilun,POS=21)p%boxlen
    read(ilun,POS=29)noutput
    skip=4*(11+4*noutput)+1
    read(ilun,POS=skip)p%t
    skip=skip+4*(2+4*p%nlevelmax+2+2*17)
    read(ilun,POS=skip)p%gamma
    close(ilun)
    p%ncpu=nfile1
  end subroutine read_ramses_params

  subroutine read_part_header
    character(len=128)::fn,fields
    integer::ilun
    fn=trim(repository)//'/'//trim(prefix)//'_header.txt'
    ilun=12
    open(unit=ilun,file=fn,form='formatted')
    read(ilun,*)
    read(ilun,*)npart
    read(ilun,*)
    read(ilun,*)nfile
    read(ilun,*)
    read(ilun,*)fields
    close(ilun)
  end subroutine read_part_header

  subroutine read_info
    character(len=128)::fn
    character(len=80)::GMGM
    integer::ilun,nfile1,ncpu1,ndim1,levelmin1,levelmax1
    real(kind=8)::boxlen,t,omega_m,omega_b,omega_l,omega_k,gamma,h0
    fn=trim(repository)//'/info.txt'
    ilun=13
    open(unit=ilun,file=fn,form='formatted')
    read(ilun,'(A13,I11)')GMGM,nfile1
    read(ilun,'(A13,I11)')GMGM,ncpu1
    read(ilun,'(A13,I11)')GMGM,ndim1
    read(ilun,'(A13,I11)')GMGM,levelmin1
    read(ilun,'(A13,I11)')GMGM,levelmax1
    read(ilun,*)
    read(ilun,*)
    read(ilun,*)
    read(ilun,'(A13,E23.15)')GMGM,boxlen
    read(ilun,'(A13,E23.15)')GMGM,t
    read(ilun,'(A13,E23.15)')GMGM,p%texp
    read(ilun,'(A13,E23.15)')GMGM,p%aexp
    read(ilun,'(A13,E23.15)')GMGM,h0
    read(ilun,'(A13,E23.15)')GMGM,omega_m
    read(ilun,'(A13,E23.15)')GMGM,omega_l
    read(ilun,'(A13,E23.15)')GMGM,omega_k
    read(ilun,'(A13,E23.15)')GMGM,omega_b
    read(ilun,'(A13,E23.15)')GMGM,gamma
    read(ilun,'(A13,E23.15)')GMGM,p%unit_l
    read(ilun,'(A13,E23.15)')GMGM,p%unit_d
    read(ilun,'(A13,E23.15)')GMGM,p%unit_t
    read(ilun,*)
    close(ilun)
  end subroutine read_info

  subroutine deposit_ngp3d(cube_arr,nxg,nyg,nzg,ddxg,ddyg,ddzg,mpartg,periodic_g)
    real(kind=8),dimension(0:nxg,0:nyg,0:nzg)::cube_arr
    integer::nxg,nyg,nzg
    real(kind=8)::ddxg,ddyg,ddzg,mpartg
    logical::periodic_g
    integer::ix,iy,iz
    real(kind=8)::ddx,ddy,ddz
    ! Apply -0.5 shift (same centering as ramses part2cube)
    ddx = ddxg - 0.5d0
    ddy = ddyg - 0.5d0
    ddz = ddzg - 0.5d0
    ! Nearest grid point
    ix=nint(ddx)
    iy=nint(ddy)
    iz=nint(ddz)
    if(periodic_g)then
       if(ix<0)ix=ix+nxg
       if(ix>=nxg)ix=ix-nxg
       if(iy<0)iy=iy+nyg
       if(iy>=nyg)iy=iy-nyg
       if(iz<0)iz=iz+nzg
       if(iz>=nzg)iz=iz-nzg
       if(ix>=0.and.ix<nxg.and.iy>=0.and.iy<nyg.and.iz>=0.and.iz<nzg)then
          cube_arr(ix,iy,iz)=cube_arr(ix,iy,iz)+mpartg
       endif
    else
       if(ix>=0.and.ix<=nxg.and.iy>=0.and.iy<=nyg.and.iz>=0.and.iz<=nzg)then
          cube_arr(ix,iy,iz)=cube_arr(ix,iy,iz)+mpartg
       endif
    endif
  end subroutine deposit_ngp3d

  subroutine deposit_cic3d(cube_arr,nxg,nyg,nzg,ddxg,ddyg,ddzg,mpartg,periodic_g)
    real(kind=8),dimension(0:nxg,0:nyg,0:nzg)::cube_arr
    integer::nxg,nyg,nzg
    real(kind=8)::ddxg,ddyg,ddzg,mpartg
    logical::periodic_g
    integer::ix,iy,iz,ixp1,iyp1,izp1
    real(kind=8)::fx,fy,fz
    real(kind=8)::ddx,ddy,ddz
    ! Apply -0.5 shift with +1 safety offset (same centering as ramses part2cube)
    ddx = ddxg - 0.5d0 + 1.0d0
    ddy = ddyg - 0.5d0 + 1.0d0
    ddz = ddzg - 0.5d0 + 1.0d0
    ix=int(ddx)
    iy=int(ddy)
    iz=int(ddz)
    ! Remove safety offset
    ix = ix - 1
    iy = iy - 1
    iz = iz - 1
    ddx = ddx - 1.0d0
    ddy = ddy - 1.0d0
    ddz = ddz - 1.0d0
    ! Fractional position within cell
    fx = ddx - dble(ix)
    fy = ddy - dble(iy)
    fz = ddz - dble(iz)
    if(periodic_g)then
       if(ix<0)ix=ix+nxg
       if(ix>=nxg)ix=ix-nxg
       if(iy<0)iy=iy+nyg
       if(iy>=nyg)iy=iy-nyg
       if(iz<0)iz=iz+nzg
       if(iz>=nzg)iz=iz-nzg
    endif
    ixp1=ix+1
    iyp1=iy+1
    izp1=iz+1
    if(periodic_g)then
       if(ixp1<0)ixp1=ixp1+nxg
       if(ixp1>=nxg)ixp1=ixp1-nxg
       if(iyp1<0)iyp1=iyp1+nyg
       if(iyp1>=nyg)iyp1=iyp1-nyg
       if(izp1<0)izp1=izp1+nzg
       if(izp1>=nzg)izp1=izp1-nzg
    endif
    if(ix>=0.and.ix<nxg.and.iy>=0.and.iy<nyg.and.iz>=0.and.iz<nzg &
 &   .and.fx>=0.and.fy>=0.and.fz>=0)then
       cube_arr(ix  ,iy  ,iz  )=cube_arr(ix  ,iy  ,iz  )+mpartg*(1d0-fx)*(1d0-fy)*(1d0-fz)
       cube_arr(ix  ,iy  ,izp1)=cube_arr(ix  ,iy  ,izp1)+mpartg*(1d0-fx)*(1d0-fy)*fz
       cube_arr(ix  ,iyp1,iz  )=cube_arr(ix  ,iyp1,iz  )+mpartg*(1d0-fx)*fy*(1d0-fz)
       cube_arr(ix  ,iyp1,izp1)=cube_arr(ix  ,iyp1,izp1)+mpartg*(1d0-fx)*fy*fz
       cube_arr(ixp1,iy  ,iz  )=cube_arr(ixp1,iy  ,iz  )+mpartg*fx*(1d0-fy)*(1d0-fz)
       cube_arr(ixp1,iy  ,izp1)=cube_arr(ixp1,iy  ,izp1)+mpartg*fx*(1d0-fy)*fz
       cube_arr(ixp1,iyp1,iz  )=cube_arr(ixp1,iyp1,iz  )+mpartg*fx*fy*(1d0-fz)
       cube_arr(ixp1,iyp1,izp1)=cube_arr(ixp1,iyp1,izp1)+mpartg*fx*fy*fz
    endif
  end subroutine deposit_cic3d

  subroutine deposit_tsc3d(cube_arr,nxg,nyg,nzg,ddxg,ddyg,ddzg,mpartg,periodic_g)
    real(kind=8),dimension(0:nxg,0:nyg,0:nzg)::cube_arr
    integer::nxg,nyg,nzg
    real(kind=8)::ddxg,ddyg,ddzg,mpartg
    logical::periodic_g
    integer::ixc,iyc,izc,ix,iy,iz,dxi,dyi,dzi
    real(kind=8)::xc,yc,zc,wx(3),wy(3),wz(3),dxrel,dyrel,dzrel
    real(kind=8)::ddx,ddy,ddz
    integer::ix_idx,iy_idx,iz_idx

    ! Apply -0.5 shift (same centering as ramses part2cube)
    ddx = ddxg - 0.5d0
    ddy = ddyg - 0.5d0
    ddz = ddzg - 0.5d0
    ixc=int(ddx+0.5d0); iyc=int(ddy+0.5d0); izc=int(ddz+0.5d0)
    xc=dble(ixc)+0.5d0
    yc=dble(iyc)+0.5d0
    zc=dble(izc)+0.5d0
    dxrel=ddx-xc
    dyrel=ddy-yc
    dzrel=ddz-zc

    wx(1)=0.5d0*(1.5d0-abs(dxrel+1.0d0))**2
    wx(2)=0.75d0-        (dxrel        )**2
    wx(3)=0.5d0*(1.5d0-abs(dxrel-1.0d0))**2

    wy(1)=0.5d0*(1.5d0-abs(dyrel+1.0d0))**2
    wy(2)=0.75d0-        (dyrel        )**2
    wy(3)=0.5d0*(1.5d0-abs(dyrel-1.0d0))**2

    wz(1)=0.5d0*(1.5d0-abs(dzrel+1.0d0))**2
    wz(2)=0.75d0-        (dzrel        )**2
    wz(3)=0.5d0*(1.5d0-abs(dzrel-1.0d0))**2

    do dxi=-1,1
       do dyi=-1,1
          do dzi=-1,1
             ix=ixc+dxi
             iy=iyc+dyi
             iz=izc+dzi
             ix_idx=dxi+2
             iy_idx=dyi+2
             iz_idx=dzi+2
             if(periodic_g)then
                if(ix<0)ix=ix+nxg
                if(ix>=nxg)ix=ix-nxg
                if(iy<0)iy=iy+nyg
                if(iy>=nyg)iy=iy-nyg
                if(iz<0)iz=iz+nzg
                if(iz>=nzg)iz=iz-nzg
                if(ix>=0.and.ix<nxg.and.iy>=0.and.iy<nyg.and.iz>=0.and.iz<nzg)then
                   cube_arr(ix,iy,iz)=cube_arr(ix,iy,iz)+mpartg*wx(ix_idx)*wy(iy_idx)*wz(iz_idx)
                endif
             else
                if(ix>=0.and.ix<=nxg.and.iy>=0.and.iy<=nyg.and.iz>=0.and.iz<=nzg)then
                   cube_arr(ix,iy,iz)=cube_arr(ix,iy,iz)+mpartg*wx(ix_idx)*wy(iy_idx)*wz(iz_idx)
                endif
             endif
          end do
       end do
    end do
  end subroutine deposit_tsc3d

  subroutine deposit_pcs3d(cube_arr,nxg,nyg,nzg,ddxg,ddyg,ddzg,mpartg,periodic_g)
    real(kind=8),dimension(0:nxg,0:nyg,0:nzg)::cube_arr
    integer::nxg,nyg,nzg
    real(kind=8)::ddxg,ddyg,ddzg,mpartg
    logical::periodic_g
    integer::ixc,iyc,izc,ix,iy,iz,dxi,dyi,dzi
    real(kind=8)::wx(4),wy(4),wz(4)
    real(kind=8)::dxrel,dyrel,dzrel
    real(kind=8)::ddx,ddy,ddz
    integer::ix_idx,iy_idx,iz_idx

    ! Apply -0.5 shift (same centering as ramses part2cube)
    ddx = ddxg - 0.5d0
    ddy = ddyg - 0.5d0
    ddz = ddzg - 0.5d0
    ixc=int(ddx); iyc=int(ddy); izc=int(ddz)
    dxrel=ddx-(dble(ixc)+0.5d0)
    dyrel=ddy-(dble(iyc)+0.5d0)
    dzrel=ddz-(dble(izc)+0.5d0)

    wx(1)=(2d0-abs(dxrel+1.5d0))**3/6d0
    wx(2)=(4d0-6d0*(dxrel+0.5d0)**2+3d0*abs(dxrel+0.5d0)**3)/6d0
    wx(3)=(4d0-6d0*(dxrel-0.5d0)**2+3d0*abs(dxrel-0.5d0)**3)/6d0
    wx(4)=(2d0-abs(dxrel-1.5d0))**3/6d0

    wy(1)=(2d0-abs(dyrel+1.5d0))**3/6d0
    wy(2)=(4d0-6d0*(dyrel+0.5d0)**2+3d0*abs(dyrel+0.5d0)**3)/6d0
    wy(3)=(4d0-6d0*(dyrel-0.5d0)**2+3d0*abs(dyrel-0.5d0)**3)/6d0
    wy(4)=(2d0-abs(dyrel-1.5d0))**3/6d0

    wz(1)=(2d0-abs(dzrel+1.5d0))**3/6d0
    wz(2)=(4d0-6d0*(dzrel+0.5d0)**2+3d0*abs(dzrel+0.5d0)**3)/6d0
    wz(3)=(4d0-6d0*(dzrel-0.5d0)**2+3d0*abs(dzrel-0.5d0)**3)/6d0
    wz(4)=(2d0-abs(dzrel-1.5d0))**3/6d0

    do dxi=-2,1
       do dyi=-2,1
          do dzi=-2,1
             ix=ixc+dxi
             iy=iyc+dyi
             iz=izc+dzi
             ix_idx=dxi+3
             iy_idx=dyi+3
             iz_idx=dzi+3
             if(periodic_g)then
                if(ix<0)ix=ix+nxg
                if(ix>=nxg)ix=ix-nxg
                if(iy<0)iy=iy+nyg
                if(iy>=nyg)iy=iy-nyg
                if(iz<0)iz=iz+nzg
                if(iz>=nzg)iz=iz-nzg
                if(ix>=0.and.ix<nxg.and.iy>=0.and.iy<nyg.and.iz>=0.and.iz<nzg)then
                   cube_arr(ix,iy,iz)=cube_arr(ix,iy,iz)+mpartg*wx(ix_idx)*wy(iy_idx)*wz(iz_idx)
                endif
             else
                if(ix>=0.and.ix<=nxg.and.iy>=0.and.iy<=nyg.and.iz>=0.and.iz<=nzg)then
                   cube_arr(ix,iy,iz)=cube_arr(ix,iy,iz)+mpartg*wx(ix_idx)*wy(iy_idx)*wz(iz_idx)
                endif
             endif
          end do
       end do
    end do
  end subroutine deposit_pcs3d

end program part2cube

function check_ramses_exist(repository,prefix)
  logical::check_ramses_exist
  character(len=80)::repository,prefix
  character(len=128)::nomfich_part
  check_ramses_exist=.true.
  nomfich_part=trim(repository)//'/'//trim(prefix)//'.00001'
  inquire(file=nomfich_part, exist=check_ramses_exist)
end function check_ramses_exist

subroutine title(n,nchar)
  implicit none
  integer::n
  character*5::nchar
  character*1::nchar1
  character*2::nchar2
  character*3::nchar3
  character*4::nchar4
  character*5::nchar5
  if(n.ge.10000)then
     write(nchar5,'(i5)') n
     nchar = nchar5
  elseif(n.ge.1000)then
     write(nchar4,'(i4)') n
     nchar = '0'//nchar4
  elseif(n.ge.100)then
     write(nchar3,'(i3)') n
     nchar = '00'//nchar3
  elseif(n.ge.10)then
     write(nchar2,'(i2)') n
     nchar = '000'//nchar2
  else
     write(nchar1,'(i1)') n
     nchar = '0000'//nchar1
  endif
end subroutine title


