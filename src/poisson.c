// This is the explicit conjugate gradient method for descrete Puasson problem
// on nonuniform mesh.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Domain size.
const double A = 2.0;
const double B = 2.0;

int NX, NY;                                 // the number of internal points on axes (ox) and (oy).
double * XNodes, * YNodes;						// mesh node coords are stored here.

#define Test
#define Print
#define Step 10

#define Max(A,B) ((A)>(B)?(A):(B))
#define R2(x,y) ((x)*(x)+(y)*(y))
#define Cube(x) ((x)*(x)*(x))

#define hx(i)  (XNodes[i+1]-XNodes[i])
#define hy(j)  (YNodes[j+1]-YNodes[j])

#define LeftPart(P,i,j)\
((-(P[NX*(j)+i+1]-P[NX*(j)+i])/hx(i)+(P[NX*(j)+i]-P[NX*(j)+i-1])/hx(i-1))/(0.5*(hx(i)+hx(i-1)))+\
(-(P[NX*(j+1)+i]-P[NX*(j)+i])/hy(j)+(P[NX*(j)+i]-P[NX*(j-1)+i])/hy(j-1))/(0.5*(hy(j)+hy(j-1))))

#define TRUE  ((int) 1)
#define FALSE ((int) 0)

int IsPower(int Number)
// the function returns log_{2}(Number) if it is integer. If not it returns (-1).
{
    unsigned int M;
    int p;

    if(Number <= 0)
        return(-1);

    M = Number; p = 0;
    while(M % 2 == 0)
    {
        ++p;
        M = M >> 1;
    }
    if((M >> 1) != 0)
        return(-1);
    else
        return(p);

}

int SplitFunction(int N0, int N1, int p)
// This is the splitting procedure of proc. number p. The integer p0
// is calculated such that abs(N0/p0 - N1/(p-p0)) --> min.
{
    float n0, n1;
    int p0, i;

    n0 = (float) N0; n1 = (float) N1;
    p0 = 0;

    for(i = 0; i < p; i++)
        if(n0 > n1)
        {
            n0 = n0 / 2.0;
            ++p0;
        }
        else
            n1 = n1 / 2.0;

    return(p0);
}

double Solution(double x,double y)
{
//    if (x < 0.0000001 || y < 0.0000001)
//        return exp(1.0);
    return exp(1.0-x*x*y*y);
}

double BoundaryValue(double x, double y)
{
    return Solution(x,y);
}

int RightPart(double * rhs)
{
    int i, j;

    memset(rhs,0,NX*NY*sizeof(double));
    for(j=0; j<NY; j++)
        for(i=0; i<NX; i++)
            rhs[j*NX+i] = 2.0 * R2(XNodes[i], YNodes[j]) * (1.0 - 2.0 * XNodes[i] * XNodes[i] * YNodes[j] * YNodes[j]) * exp(1.0 - XNodes[i] * XNodes[i] * YNodes[j] * YNodes[j]);
        return 0;
}

int MeshGenerate(int NX, int NY)
{
    const double q = 1.5;
    int i;

    for(i=0; i<NX; i++)
        XNodes[i] = A*(pow(1.0+i/(NX-1.0),q)-1.0)/(pow(2.0,q)-1.0);
    for(i=0; i<NY; i++)
        YNodes[i] = B*(pow(1.0+i/(NY-1.0),q)-1.0)/(pow(2.0,q)-1.0);
    return 0;
}

int main(int argc, char * argv[])
{
    double * SolVect;                       // the solution array
    double * ResVect;                       // the residual array
    double * BasisVect;                     // the vector of A-orthogonal system in CGM
    double * RHS_Vect;                      // the right hand side of Puasson equation.
    double * buff;
    double sp, alpha, tau, NewValue, err;   // auxiliary values
    int SDINum, CGMNum;                     // the number of steep descent and CGM iterations.
    int counter;                            // the current iteration number.

    int i,j;
    char str[127];
    FILE * fp;

    int ProcNum, rank;              // the number of processes and rank in communicator.
    int power, p0, p1;              // ProcNum = 2^(power), power splits into sum p0 + p1.
    int dims[2];                    // dims[0] = 2^p0, dims[1] = 2^p1 (--> M = dims[0]*dims[1]).
    int n0,n1, k0,k1;               // N0 = n0*dims[0] + k0, N1 = n1*dims[1] + k1.
    int Coords[2];                  // the process coordinates in the cartesian topology created for mesh.

    MPI_Comm Grid_Comm;             // this is a handler of a new communicator.
    const int ndims = 2;            // the number of a process topology dimensions.
    int periods[2] = {0,0};         // it is used for creating processes topology.
    int left, right, up, down;      // the neighbours of the process.

    // MPI Library is being activated ...
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    // command line analizer
    switch (argc)
    {
    case 4:{
                SDINum = 1;
                CGMNum = atoi(argv[3]);
                break;
           }
    case 5:{
                SDINum = Max(atoi(argv[3]),1);      // SDINum >= 1
                CGMNum = atoi(argv[4]);
                break;
           }
    default:{
                if (rank == 0)
                    printf("Wrong number of parameters in command line.\nUsage: <ProgName> "
                           "<Nodes number on (0x) axis> <Nodes number on (0y) axis> "
                           "[the number of steep descent iterations] "
                           "<the number of conjugate gragient iterations>\nFinishing...\n");
                return(-1);
            }
    }

    NX = atoi(argv[1]); NY = atoi(argv[2]);

    if((NX <= 0)||(NY <= 0))
    {
        if(rank == 0)
            printf("The first and the second arguments (mesh numbers) should be positive.\n");

        MPI_Finalize();
        return(2);
    }

    if((power = IsPower(ProcNum)) < 0)
    {
        if(rank == 0)
            printf("The number of procs must be a power of 2.\n");
        MPI_Finalize();
        return(3);
    }

    p0 = SplitFunction(NX, NY, power);
    p1 = power - p0;

    dims[0] = (unsigned int) 1 << p0;   dims[1] = (unsigned int) 1 << p1;
    n0 = NX >> p0;                      n1 = NY >> p1;
    k0 = NX - dims[0]*n0;               k1 = NY - dims[1]*n1;

#ifdef Print
    if(rank == 0)
    {
        printf("The number of processes ProcNum = 2^%d. It is split into %d x %d processes.\n"
               "The number of nodes N0 = %d, N1 = %d. Blocks B(i,j) have size:\n", power, dims[0],dims[1], NX,NY);

        if((k0 > 0)&&(k1 > 0))
            printf("-->\t %d x %d iff i = 0 .. %d, j = 0 .. %d;\n", n0+1,n1+1, k0-1,k1-1);
        if(k1 > 0)
            printf("-->\t %d x %d iff i = %d .. %d, j = 0 .. %d;\n", n0,n1+1, k0,dims[0]-1, k1-1);
        if(k0 > 0)
            printf("-->\t %d x %d iff i = 0 .. %d, j = %d .. %d;\n", n0+1,n1, k0-1, k1,dims[1]-1);

        printf("-->\t %d x %d iff i = %d .. %d, j = %d .. %d.\n", n0,n1, k0,dims[0]-1, k1,dims[1]-1);
    }
#endif

    // the cartesian topology of processes is being created ...
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, TRUE, &Grid_Comm);
    MPI_Comm_rank(Grid_Comm, &rank);
    MPI_Cart_coords(Grid_Comm, rank, ndims, Coords);

    if(Coords[0] < k0)
        ++n0;
    if(Coords[1] < k1)
        ++n1;

    int sX, fX, sY, fY;

    if (Coords[0] < k0)
        sX = n0*Coords[0];
    else
        sX = (n0+1)*k0 + n0*(Coords[0]-k0);

    if (Coords[1] < k1)
        sY = n1*Coords[1];
    else
        sY = (n1+1)*k1 + n1*(Coords[1]-k1);

    fX = sX + n0;
    fY = sY + n1;

    if (Coords[0] == 0)
        sX = 1;
    if (Coords[1] == 0)
        sY = 1;
    if (Coords[0] == dims[0]-1)
        fX--;
    if (Coords[1] == dims[1]-1)
        fY--;

    MPI_Cart_shift(Grid_Comm, 0, 1, &left, &right);
    MPI_Cart_shift(Grid_Comm, 1, 1, &down, &up);

#ifdef Print
    printf("!!! Coords: (%d, %d), starts: %d %d   fns: %d %d\n", Coords[0], Coords[1], sX, sY, fX, fY);
    printf("My Rank in Grid_Comm is %d. My topological coords is (%d,%d). Domain size is %d x %d nodes.\n"
           "My neighbours: left = %d, right = %d, down = %d, up = %d.\n",
           rank, Coords[0], Coords[1], n0, n1, left,right, down,up);
#endif

    if (rank == 0)
    {
        sprintf(str,"PuassonSerial_ECGM_%dx%d.log", NX, NY);
        fp = fopen(str,"w");
        fprintf(fp,"The Domain: [0,%f]x[0,%f], number of points: N[0,A] = %d, N[0,B] = %d;\n"
                   "The steep descent iterations number: %d\n"
                   "The conjugate gradient iterations number: %d\n",
                    A,B, NX,NY, SDINum,CGMNum);
    }

    XNodes = (double *)malloc(NX*sizeof(double));
    YNodes = (double *)malloc(NY*sizeof(double));
    SolVect   = (double *)malloc(NX*NY*sizeof(double));
    ResVect   = (double *)malloc(NX*NY*sizeof(double));
    RHS_Vect  = (double *)malloc(NX*NY*sizeof(double));
    buff  = (double *)malloc(NY*sizeof(double));

// Initialization of Arrays
    MeshGenerate(NX, NY);
    memset(ResVect,0,NX*NY*sizeof(double));
    memset(SolVect,0,NX*NY*sizeof(double));
    RightPart(RHS_Vect);

    for(i=0; i<NX; i++)
    {
        SolVect[i] = BoundaryValue(XNodes[i],0.0);
        SolVect[NX*(NY-1)+i] = BoundaryValue(XNodes[i],B);
    }
    for(j=0; j<NY; j++)
    {
        SolVect[NX*j] = BoundaryValue(0.0,YNodes[j]);
        SolVect[NX*j+(NX-1)] = BoundaryValue(A,YNodes[j]);
    }

// Iterations ...
    #ifdef Test
        err = 0.0;
        for(j=1; j < NY-1; j++)
            for(i=1; i < NX-1; i++)
                err = Max(err, fabs(Solution(XNodes[i],YNodes[j])-SolVect[NX*j+i]));
        if (rank == 0)
            fprintf(fp,"\nNo iterations have been performed. The residual error is %.12f\n", err);
    #endif

// Steep descent iterations begin ...
    #ifdef Print
        if (rank == 0)
            printf("\nSteep descent iterations begin ...\n");
    #endif

    for(counter=1; counter<=SDINum; counter++)
    {
// The residual vector r(k) = Ax(k)-f is calculating ...
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
                ResVect[NX*j+i] = LeftPart(SolVect,i,j)-RHS_Vect[NX*j+i];

// ResVect synchronization
        //send..
        if (Coords[0] > 0)
        {
            for(j=sY; j < fY; j++)
                buff[j] = ResVect[NX*j+sX];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, left, 4, Grid_Comm);
        }
        if (Coords[0] < (dims[0]-1))
        {
            for(j=sY; j < fY; j++)
                buff[j] = ResVect[NX*j+(fX-1)];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, right, 1, Grid_Comm);
        }
        if (Coords[1] > 0)
        {
            MPI_Send(&ResVect[NX*(sY)+sX], n0, MPI_DOUBLE, down, 2, Grid_Comm);
        }
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Send(&ResVect[NX*(fY-1)+sX], n0, MPI_DOUBLE, up, 3, Grid_Comm);
        }

        //..and receive
        if (Coords[1] < (dims[1]-1)) // from up
        {
            MPI_Recv(&ResVect[NX*(fY)+sX], n0, MPI_DOUBLE, up, 2, Grid_Comm, NULL);
        }
        if (Coords[1] > 0) // from down
        {
            MPI_Recv(&ResVect[NX*(sY-1)+sX], n0, MPI_DOUBLE, down, 3, Grid_Comm, NULL);
        }
        if (Coords[0] > 0) // from left
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, left, 1, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                ResVect[NX*j+(sX-1)] = buff[j];
        }
        if (Coords[0] < (dims[0]-1)) // from right
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, right, 4, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                ResVect[NX*j+fX] = buff[j];
        }

// The value of product (r(k),r(k)) is calculating ...
        sp = 0.0;
        double local_sp = 0.0;
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
                local_sp += ResVect[NX*j+i]*ResVect[NX*j+i]*hx(i)*hy(j);
        MPI_Allreduce(&local_sp, &sp, 1, MPI_DOUBLE, MPI_SUM, Grid_Comm);
        tau = sp;

// The value of product sp = (Ar(k),r(k)) is calculating ...
        local_sp = 0.0;
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
                local_sp += LeftPart(ResVect,i,j)*ResVect[NX*j+i]*hx(i)*hy(j);
        MPI_Allreduce(&local_sp, &sp, 1, MPI_DOUBLE, MPI_SUM, Grid_Comm);
        tau = tau/sp;

// The x(k+1) is calculating ...
        err = 0.0;
        double local_err = 0.0;
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
            {
                NewValue = SolVect[NX*j+i]-tau*ResVect[NX*j+i];
                local_err = Max(local_err, fabs(NewValue-SolVect[NX*j+i]));
                SolVect[NX*j+i] = NewValue;
            }
        MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, Grid_Comm);

// SolVect synchronization
        //send..
        if (Coords[0] > 0)
        {
            for(j=sY; j < fY; j++)
                buff[j] = SolVect[NX*j+sX];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, left, 4, Grid_Comm);
        }
        if (Coords[0] < (dims[0]-1))
        {
            for(j=sY; j < fY; j++)
                buff[j] = SolVect[NX*j+(fX-1)];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, right, 1, Grid_Comm);
        }
        if (Coords[1] > 0)
        {
            MPI_Send(&SolVect[NX*(sY)+sX], n0, MPI_DOUBLE, down, 2, Grid_Comm);
        }
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Send(&SolVect[NX*(fY-1)+sX], n0, MPI_DOUBLE, up, 3, Grid_Comm);
        }

        //..and receive
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Recv(&SolVect[NX*(fY)+sX], n0, MPI_DOUBLE, up, 2, Grid_Comm, NULL);
        }
        if (Coords[1] > 0)
        {
            MPI_Recv(&SolVect[NX*(sY-1)+sX], n0, MPI_DOUBLE, down, 3, Grid_Comm, NULL);
        }
        if (Coords[0] > 0)
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, left, 1, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                SolVect[NX*j+(sX-1)] = buff[j];
        }
        if (Coords[0] < (dims[0]-1))
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, right, 4, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                SolVect[NX*j+fX] = buff[j];
        }

        if(counter%Step == 0)
        {
            if (rank == 0)
                printf("The Steep Descent iteration %d has been performed.\n",counter);

    #ifdef Print
            if (rank == 0)
                fprintf(fp,"\nThe Steep Descent iteration k = %d has been performed.\n"
                        "Step \\tau(k) = %f.\nThe difference value is estimated by %.12f.\n",\
                        counter, tau, err);
    #endif

    #ifdef Test
            err = 0.0;
            double local_err = 0.0;
            for(j=sY; j < fY; j++)
                for(i=sX; i < fX; i++)
                    local_err = Max(local_err, fabs(Solution(XNodes[i],YNodes[j])-SolVect[NX*j+i]));
            MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, Grid_Comm);

            if (rank == 0)
                fprintf(fp,"The Steep Descent iteration %d have been performed. "
                        "The residual error is %.12f\n", counter, err);
    #endif
        }
    }
// the end of steep descent iteration.

    BasisVect = ResVect;    // g(0) = r(k-1).
    ResVect = (double *)malloc(NX*NY*sizeof(double));
    memset(ResVect,0,NX*NY*sizeof(double));

// CGM iterations begin ...
// sp == (Ar(k-1),r(k-1)) == (Ag(0),g(0)), k=1.
    #ifdef Print
        if (rank == 0)
            printf("\nCGM iterations begin ...\n");
    #endif

    for(counter=1; counter<=CGMNum; counter++)
    {
    // The residual vector r(k) is calculating ...
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
                ResVect[NX*j+i] = LeftPart(SolVect,i,j)-RHS_Vect[NX*j+i];

    // ResVect synchronization
        //send..
        if (Coords[0] > 0)
        {
            for(j=sY; j < fY; j++)
                buff[j] = ResVect[NX*j+sX];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, left, 4, Grid_Comm);
        }
        if (Coords[0] < (dims[0]-1))
        {
            for(j=sY; j < fY; j++)
                buff[j] = ResVect[NX*j+(fX-1)];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, right, 1, Grid_Comm);
        }
        if (Coords[1] > 0)
        {
            MPI_Send(&ResVect[NX*(sY)+sX], n0, MPI_DOUBLE, down, 2, Grid_Comm);
        }
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Send(&ResVect[NX*(fY-1)+sX], n0, MPI_DOUBLE, up, 3, Grid_Comm);
        }

        //..and receive
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Recv(&ResVect[NX*(fY)+sX], n0, MPI_DOUBLE, up, 2, Grid_Comm, NULL);
        }
        if (Coords[1] > 0)
        {
            MPI_Recv(&ResVect[NX*(sY-1)+sX], n0, MPI_DOUBLE, down, 3, Grid_Comm, NULL);
        }
        if (Coords[0] > 0)
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, left, 1, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                ResVect[NX*j+(sX-1)] = buff[j];
        }
        if (Coords[0] < (dims[0]-1))
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, right, 4, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                ResVect[NX*j+fX] = buff[j];
        }

    // The value of product (Ar(k),g(k-1)) is calculating ...
        alpha = 0.0;
        double local_alpha = 0.0;
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
                local_alpha += LeftPart(ResVect,i,j)*BasisVect[NX*j+i]*hx(i)*hy(j);
        MPI_Allreduce(&local_alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, Grid_Comm);
        alpha = alpha/sp;

    // The new basis vector g(k) is being calculated ...
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
                BasisVect[NX*j+i] = ResVect[NX*j+i]-alpha*BasisVect[NX*j+i];
    // BasisVect synchronization
        //send..
        if (Coords[0] > 0)
        {
            for(j=sY; j < fY; j++)
                buff[j] = BasisVect[NX*j+sX];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, left, 4, Grid_Comm);
        }
        if (Coords[0] < (dims[0]-1))
        {
            for(j=sY; j < fY; j++)
                buff[j] = BasisVect[NX*j+(fX-1)];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, right, 1, Grid_Comm);
        }
        if (Coords[1] > 0)
        {
            MPI_Send(&BasisVect[NX*(sY)+sX], n0, MPI_DOUBLE, down, 2, Grid_Comm);
        }
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Send(&BasisVect[NX*(fY-1)+sX], n0, MPI_DOUBLE, up, 3, Grid_Comm);
        }

        //..and receive
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Recv(&BasisVect[NX*(fY)+sX], n0, MPI_DOUBLE, up, 2, Grid_Comm, NULL);
        }
        if (Coords[1] > 0)
        {
            MPI_Recv(&BasisVect[NX*(sY-1)+sX], n0, MPI_DOUBLE, down, 3, Grid_Comm, NULL);
        }
        if (Coords[0] > 0)
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, left, 1, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                BasisVect[NX*j+(sX-1)] = buff[j];
        }
        if (Coords[0] < (dims[0]-1))
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, right, 4, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                BasisVect[NX*j+fX] = buff[j];
        }

    // The value of product (r(k),g(k)) is being calculated ...
        tau = 0.0;
        double local_tau = 0.0;
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
                local_tau += ResVect[NX*j+i]*BasisVect[NX*j+i]*hx(i)*hy(j);
        MPI_Allreduce(&local_tau, &tau, 1, MPI_DOUBLE, MPI_SUM, Grid_Comm);

    // The value of product sp = (Ag(k),g(k)) is being calculated ...
        sp = 0.0;
        double local_sp = 0.0;
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
                local_sp += LeftPart(BasisVect,i,j)*BasisVect[NX*j+i]*hx(i)*hy(j);
        MPI_Allreduce(&local_sp, &sp, 1, MPI_DOUBLE, MPI_SUM, Grid_Comm);
        tau = tau/sp;

    // The x(k+1) is being calculated ...
        err = 0.0;
        double local_err = 0.0;
        for(j=sY; j < fY; j++)
            for(i=sX; i < fX; i++)
            {
                NewValue = SolVect[NX*j+i]-tau*BasisVect[NX*j+i];
                local_err = Max(local_err, fabs(NewValue-SolVect[NX*j+i]));
                SolVect[NX*j+i] = NewValue;
            }
        MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, Grid_Comm);

    // SolVect synchronization
        //send..
        if (Coords[0] > 0)
        {
            for(j=sY; j < fY; j++)
                buff[j] = SolVect[NX*j+sX];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, left, 4, Grid_Comm);
        }
        if (Coords[0] < (dims[0]-1))
        {
            for(j=sY; j < fY; j++)
                buff[j] = SolVect[NX*j+(fX-1)];
            MPI_Send(&buff[sY], n1, MPI_DOUBLE, right, 1, Grid_Comm);
        }
        if (Coords[1] > 0)
        {
            MPI_Send(&SolVect[NX*(sY)+sX], n0, MPI_DOUBLE, down, 2, Grid_Comm);
        }
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Send(&SolVect[NX*(fY-1)+sX], n0, MPI_DOUBLE, up, 3, Grid_Comm);
        }

        //..and receive
        if (Coords[1] < (dims[1]-1))
        {
            MPI_Recv(&SolVect[NX*(fY)+sX], n0, MPI_DOUBLE, up, 2, Grid_Comm, NULL);
        }
        if (Coords[1] > 0)
        {
            MPI_Recv(&SolVect[NX*(sY-1)+sX], n0, MPI_DOUBLE, down, 3, Grid_Comm, NULL);
        }
        if (Coords[0] > 0)
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, left, 1, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                SolVect[NX*j+(sX-1)] = buff[j];
        }
        if (Coords[0] < (dims[0]-1))
        {
            MPI_Recv(&buff[sY], n1, MPI_DOUBLE, right, 4, Grid_Comm, NULL);
            for(j=sY; j < fY; j++)
                SolVect[NX*j+fX] = buff[j];
        }

        if(counter%Step == 0)
        {
            if (rank == 0)
                printf("The %d iteration of CGM method has been carried out.\n", counter);

#ifdef Print
            if (rank == 0)
                fprintf(fp,"\nThe iteration %d of conjugate gradient method has been finished.\n"
                       "The value of \\alpha(k) = %f, \\tau(k) = %f. The difference value is %f.\n",\
                        counter, alpha, tau, err);
#endif

#ifdef Test
            err = 0.0;
            double local_err = 0.0;
            for(j=sY; j < fY; j++)
                for(i=sX; i < fX; i++)
                    local_err = Max(local_err, fabs(Solution(XNodes[i],YNodes[j])-SolVect[NX*j+i]));
            MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, Grid_Comm);

            if (rank == 0)
                fprintf(fp,"The %d iteration of CGM have been performed. The residual error is %.12f\n",\
                        counter, err);
#endif
        }
    }
// the end of CGM iterations.

// printing some results ...
    if (rank == 0)
    {
        fprintf(fp,"\nThe %d iterations are carried out. The error of iterations is estimated by %.12f.\n",
                SDINum+CGMNum, err);
        fclose(fp);

        sprintf(str,"PuassonSerial_ECGM_%dx%d.dat", NX, NY);
        fp = fopen(str,"w");
        fprintf(fp,"# This is the conjugate gradient method for descrete Puasson equation.\n"
                "# A = %f, B = %f, N[0,A] = %d, N[0,B] = %d, SDINum = %d, CGMNum = %d.\n"
                "# One can draw it by gnuplot by the command: splot 'MyPath\\FileName.dat' with lines\n",\
                A, B, NX, NY, SDINum, CGMNum);
        for (j=0; j < NY; j++)
        {
            for (i=0; i < NX; i++)
                fprintf(fp,"\n%f %f %f", XNodes[i], YNodes[j], SolVect[NX*j+i]);
            fprintf(fp,"\n");
        }
        fclose(fp);
    }

    free(XNodes); free(YNodes);
    free(SolVect); free(ResVect); free(BasisVect); free(RHS_Vect);

    MPI_Finalize();
    // The end of MPI session ...
    return(0);
}

