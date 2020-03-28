//
//  Shaders.metal
//  noisySphere
//
//  Created by Yupeng Gu on 3/20/19.
//  Copyright © 2019 Yupeng Gu. All rights reserved.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

constant float3 lightDirection = float3(10.0,10.0,10.0);
constant float4 ambientColor = float4(0.0,0.0,0.0,1.0);
constant float4 diffuseColor = float4(0.0,0.6,0.0,1.0);






class Random{
public:
    Random(unsigned int mult = 3039177861u){_mult = mult;}
    void seed(unsigned int s){_seed = s;}
    unsigned rand(){_seed *= _mult; return _seed;}
    float uniform01(){return float(rand())/float(UINT_MAX);}
    float uniform(float minimum,float maximum){
        return minimum + (uniform01()*(maximum-minimum));
    }
    unsigned poisson(float mean){
        float g = exp(-mean);
        unsigned res = 0;
        float t = uniform01();
        while(t>g){
            res++;
            t *= uniform01();
        }
        return res;
    }
private:
    unsigned int _seed,_mult;
};

float gabor(float K,float a,float F0,float omega0,float x,float y,float tOffset){
    float gauss_env = K*exp(-M_PI_F*(a*a)*(x*x+y*y));
    float sin_carrier = cos(2.0*M_PI_F*F0*(x*cos(omega0)+y*sin(omega0)) + tOffset);
    return gauss_env*sin_carrier;
}

class Noise{
public:
    Noise(float K,float a,float F0,float omega0,float numImpulsesPerKernel,unsigned int period,unsigned int rMult = 3039177861u):K(K),a(a),F0(F0),omega0(omega0),period(period),_rMult(rMult){
        kernelRadius = sqrt(-log(0.05)/M_PI_F)/a;
        impulseDensity = numImpulsesPerKernel/(M_PI_F*kernelRadius*kernelRadius);
    }
    float cell(int i,int j,float x,float y,float tOffset){
        unsigned int s = (j%period)*period + (i%period);
        if(s==0)s=1;
        Random prng(_rMult);
        prng.seed(s);
        float impulsesPerCell = impulseDensity*kernelRadius*kernelRadius;
        unsigned int numImpulses = prng.poisson(impulsesPerCell);
        float res = 0.0;
        for(int k=0;k<(int)(numImpulses);k++){
            float xi = prng.uniform01();
            float yi = prng.uniform01();
            float wi = prng.uniform(-1.0,1.0);
            float omega0i = prng.uniform(0.0,2.0*M_PI_F);
            float xix = x-xi;
            float yiy = y-yi;
            if(xix*xix+yiy*yiy<1.0){
                res += wi*gabor(K,a,F0,omega0i,xix*kernelRadius,yiy*kernelRadius,tOffset);
            }
        }
        return res;
    }
    float gen(float x,float y,float tOffset){
        //x /= kernelRadius,y /= kernelRadius;
        float fracX = fract(x),fracY = fract(y);
        int i=int(floor(x)),j=int(floor(y));
        float noise = 0.0;
        for(int di=-1;di<=1;di++){
            for(int dj=-1;dj<=1;dj++){
                int ii = i+di,jj = j+dj;
                //if(ii<0)ii+=period;
                //if(jj<0)jj+=period;
                noise += cell(ii,jj,fracX-di,fracY-dj,tOffset);
            }
        }
        return noise;
    }
private:
    float K,a,F0,omega0,kernelRadius,impulseDensity;
    unsigned int period,_rMult;
};








typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
    float3 normal [[attribute(VertexAttributeNormal)]];
    float3 tangent [[attribute(VertexAttributeTangent)]];
    float3 bitangent [[attribute(VertexAttributeBitangent)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
    float3 eyeNormal;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> dispMap      [[ texture(TextureIndexColor) ]],
                               texture2d<half> normalMap    [[ texture(TextureIndexOutput) ]])
{
    ColorInOut out;
    
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    //float4 position = float4(in.position,1.0);
    float4 position = float4(in.position*(1.0+0.1*dispMap.sample(colorSampler,in.texCoord.xy).y), 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.texCoord = in.texCoord;
    
    half4 mapNormal = normalMap.sample(colorSampler,in.texCoord.xy)*2.0-half4(1.0,1.0,1.0,1.0);
    
    //float3 normal = in.normal;
    float3 normal = in.tangent*mapNormal.x + in.bitangent*mapNormal.y + in.normal*mapNormal.z;
    float4 eyeNormal = normalize(uniforms.normalMatrix * float4(normal,0));
    out.eyeNormal = eyeNormal.rgb;
    
    return out;
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]])
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    half4 colorSample   = colorMap.sample(colorSampler, in.texCoord.xy);
    
    float n_dot_l = max(0.0,dot(in.eyeNormal,normalize(lightDirection)));
    
    half4 lightColor = half4(ambientColor+4.0*diffuseColor*n_dot_l);
    
    colorSample.y = colorSample.y * 0.7 + 0.3;
    
    return float4(colorSample);
}

kernel void gaborNoiseKernel(constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                             texture2d<half, access::write> outTexture [[texture(TextureIndexOutput)]],
                             uint2                          gid         [[thread_position_in_grid]]){
    if((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        return;
    }
    
    Noise generator(0.5,0.05,0.02,M_PI_F/4.0,64.0,32);
    
    float intensity = 0.5+generator.gen(float(gid.x)/8.0,float(gid.y)/8.0,uniforms.tOffset);
    
    outTexture.write(half4(0.0,intensity,0.0,1.0),gid);
}

kernel void
normalKernel(texture2d<half, access::read>  inTexture  [[texture(TextureIndexColor)]],
                texture2d<half, access::write> outTexture [[texture(TextureIndexOutput)]],
                uint2                          gid         [[thread_position_in_grid]])
{
    // Check if the pixel is within the bounds of the output texture
    if((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }
    
    uint left = (gid.x - 1)%outTexture.get_width(),right = (gid.x + 1)%outTexture.get_width();
    uint down = (gid.y - 1)%outTexture.get_height(),up = (gid.y + 1)%outTexture.get_height();
    
    float3 hor = float3(1.0,0.0,(inTexture.read(uint2(right,gid.y)).y-inTexture.read(uint2(left,gid.y)).y)*8);
    float3 vert = float3(0.0,1.0,(inTexture.read(uint2(gid.x,up)).y-inTexture.read(uint2(gid.x,down)).y)*8);
    
    float4 res = float4((normalize(cross(hor,vert))+float3(1.0,1.0,1.0))/2.0,1.0);
    
    outTexture.write(half4(res),gid);
    
}

kernel void thresholdKernel(texture2d<float, access::read>  inTexture  [[texture(TextureIndexColor)]],
                            texture2d<float, access::write> outTexture [[texture(TextureIndexOutput)]],
                            uint2                          gid         [[thread_position_in_grid]]){
    if((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }
    
    float4 blank = float4(0.0,0.0,0.0,0.0);
    float4 inColor = inTexture.read(uint2(gid.x,gid.y));
    float4 outColor = inColor.y>0.995 ? inColor:blank;
    
    outTexture.write(outColor, uint2(gid.x,gid.y));
}


kernel void clearKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                        uint2 gid [[thread_position_in_grid]]){
    if((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }
    float4 black(0.0,0.0,0.0,0.0);
    outTexture.write(black, uint2(gid.x,gid.y));
}

kernel void distortNoiseKernel(texture2d<float, access::read>  inTexture [[texture(0)]],
                               texture2d<float, access::write> outTexture [[texture(1)]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               uint2 gid [[thread_position_in_grid]]){
    if((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }
    float4 inColor = inTexture.read(uint2(gid.x,gid.y));
    Noise xGen(0.5,0.05,0.02,M_PI_F/4.0,64.0,65536,3039173897u),yGen(0.5,0.05,0.02,M_PI_F/4.0,64.0,131072,3039176417u);
    float green = (1.0 + xGen.gen((float(gid.x)+uniforms.tOffset)/32.0, (float(gid.y)+uniforms.tOffset*50.0)/16.0, 0.0))/2.0;
    float blue = (1.0 + yGen.gen((float(gid.x)+uniforms.tOffset)/32.0, (float(gid.y)+uniforms.tOffset*50.0)/16.0, 0.0))/2.0;
    
    outTexture.write(float4(inColor.x,green,blue,1.0), uint2(gid.x,gid.y));
}

kernel void distortKernel(texture2d<float, access::read>  inTexture [[texture(0)]],
                          texture2d<float, access::write> outTexture [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]){
    if((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }
    float total = 0.0;
    for(int x=gid.x-7;x<=int(gid.x)+7;x++){
        for(int y=gid.y-7;y<=int(gid.y)+7;y++){
            if(x>=0 && x<int(outTexture.get_width()) && y>=0 && y<int(outTexture.get_height())){
                float4 inXY = inTexture.read(uint2(x,y));
                float intensity = inXY.x,dx0 = (inXY.y-0.5)*2.0,dy0 = (inXY.z-0.5)*2.0;
                int dx = int(7.0*dx0);
                int dy = int(7.0*dy0);
                int xx = x+dx,yy = y+dy;
                if(xx==int(gid.x) && yy==int(gid.y)){
                    total += intensity;
                }
            }
        }
    }
    
    outTexture.write(float4(0.0,log(total*3.2),0.0,0.7), uint2(gid.x,gid.y));
}


kernel void combineKernel(texture2d<float, access::read>  inTextureScreen  [[texture(0)]],
                          texture2d<float, access::read>  inTextureProc  [[texture(1)]],
                          texture2d<float, access::write> outTexture [[texture(2)]],
                          uint2                          gid         [[thread_position_in_grid]]){
    if((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }
    
    float4 inColorScreen = inTextureScreen.read(uint2(gid.x,gid.y));
    float4 inColorProc = inTextureProc.read(uint2(gid.x,gid.y));
    float4 outColor = inColorProc*4.0 + inColorScreen;
    
    outTexture.write(outColor, uint2(gid.x,gid.y));
}


kernel void combineFireKernel(texture2d<float, access::read>  inTextureScreen  [[texture(0)]],
                          texture2d<float, access::read>  inTextureProc  [[texture(1)]],
                          texture2d<float, access::write> outTexture [[texture(2)]],
                          uint2                          gid         [[thread_position_in_grid]]){
    if((gid.x >= outTexture.get_width()) || (gid.y >= outTexture.get_height()))
    {
        // Return early if the pixel is out of bounds
        return;
    }
    
    float4 outColor = inTextureScreen.read(uint2(gid.x,gid.y));
    
    int dx = (int(gid.x)-int(outTexture.get_width()/2))/6.0,dy = (int(gid.y)-int(outTexture.get_height()*7/20))/6.0;
    int xx = 256+dx,yy = 256+dy;
    
    if(xx>=0 && xx<=512 && yy>=0 && yy<=512){
        //outColor += 0.5*inTextureProc.read(uint2(uint(xx),uint(yy)));
        float4 fColor = inTextureProc.read(uint2(uint(xx),uint(yy)));
        outColor = (fColor.w*fColor + outColor.w*outColor*(1.0-fColor.w))/(fColor.w+outColor.w*(1.0-fColor.w));
    }
    
    outTexture.write(outColor, uint2(gid.x,gid.y));
}
