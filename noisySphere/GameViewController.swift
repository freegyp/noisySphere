//
//  GameViewController.swift
//  noisySphere
//
//  Created by Yupeng Gu on 3/20/19.
//  Copyright © 2019 Yupeng Gu. All rights reserved.
//

import UIKit
import MetalKit

// Our iOS specific view controller
class GameViewController: UIViewController {

    var renderer: Renderer!
    var mtkView: MTKView!
    var lastPoint = CGPoint.zero

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = view as? MTKView else {
            print("View of Gameview controller is not an MTKView")
            return
        }

        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported")
            return
        }
        
        mtkView.device = defaultDevice
        mtkView.backgroundColor = UIColor.white

        guard let newRenderer = Renderer(metalKitView: mtkView) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer

        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)

        mtkView.delegate = renderer
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)
        if let touch = touches.first as? UITouch{
            lastPoint = touch.location(in: self.view)
        }
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesMoved(touches, with: event)
        if let touch = touches.first as? UITouch{
            let currentPoint = touch.location(in: self.view)
            renderer.rotate(radians: .pi*2.0*Float(currentPoint.x-lastPoint.x)/Float(view.frame.width), axis: float3(0.0,1.0,0.0))
            lastPoint = currentPoint
        }
    }
    
    func dist(start:CGPoint,end:CGPoint)->Float{
        let d2 = (start.x-end.x)*(start.x-end.x)+(start.y-end.y)*(start.y-end.y)
        return .pi*sqrtf(Float(d2))/Float(view.frame.width)
    }
    
    func axis(start:CGPoint,end:CGPoint)->float3{
        let dx = Float(end.x-start.x)
        let dy = Float(end.y-start.y)
        return normalize(float3(dy,dx,0))
    }
}
