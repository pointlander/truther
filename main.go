// Copyright 2021 The Truther Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
)

const (
	// Size is the size of the square matrix
	Size = 5
)

var (
	// FlagNeural neural mode
	FlagNeural = flag.Bool("neural", false, "neural mode")
)

// Neural mode
func Neural(vectors *mat.CDense, values []complex128) {
	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rand.Float64()+a, (b-a)*rand.Float64()+a)
	}

	set := tc128.NewSet()
	set.Add("A", Size, Size)
	set.Add("X", Size, 1)
	set.Add("Y", Size, 1)

	w := set.Weights[0]
	for i := 0; i < cap(w.X); i++ {
		w.X = append(w.X, random128(-1, 1))
	}

	w = set.Weights[1]
	for i := 0; i < Size; i++ {
		w.X = append(w.X, vectors.At(0, i))
	}

	w = set.Weights[2]
	for i := 0; i < Size; i++ {
		w.X = append(w.X, values[0]*vectors.At(0, i))
	}

	l1 := tc128.Mul(set.Get("A"), set.Get("X"))
	cost := tc128.Quadratic(set.Get("Y"), l1)

	eta, iterations := complex128(.3), 128
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		set.Zero()

		total := tc128.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm := float64(math.Sqrt(float64(sum)))
		scaling := float64(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		w := set.Weights[0]
		for l, d := range w.D {
			w.X[l] -= eta * d * complex(scaling, 0)
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		fmt.Println(i, cmplx.Abs(total))
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			value := set.Weights[0].X[i*Size+j]
			fmt.Printf("%f ", cmplx.Abs(value))
		}
		fmt.Printf("\n")
	}
}

func main() {
	flag.Parse()
	rand.Seed(1)

	data := []float64{
		0, 1, 0, 1, 1,
		1, 0, 1, 0, 1,
		0, 1, 0, 1, 1,
		1, 0, 1, 0, 1,
		1, 1, 1, 1, 1,
	}
	adjacency := mat.NewDense(Size, Size, data)
	var eig mat.Eigen
	ok := eig.Factorize(adjacency, mat.EigenRight)
	if !ok {
		panic("Eigendecomposition failed")
	}

	values := eig.Values(nil)
	for i, value := range values {
		fmt.Println(i, value, cmplx.Abs(value), cmplx.Phase(value))
	}
	fmt.Printf("\n")

	vectors := mat.CDense{}
	eig.VectorsTo(&vectors)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			fmt.Printf("%f ", vectors.At(i, j))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")

	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			fmt.Printf("(%f, %f) ", cmplx.Abs(vectors.At(i, j)), cmplx.Phase(vectors.At(i, j)))
		}
		fmt.Printf("\n")
	}

	if *FlagNeural {
		Neural(&vectors, values)
	}

	ranks := mat.NewDense(Size, Size, nil)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			ranks.Set(i, j, real(vectors.At(i, j)))
		}
	}
	var pc stat.PC
	ok = pc.PrincipalComponents(ranks, nil)
	if !ok {
		panic("PrincipalComponents failed")
	}
	k := 2
	var proj mat.Dense
	var vec mat.Dense
	pc.VectorsTo(&vec)
	proj.Mul(ranks, vec.Slice(0, Size, 0, k))

	fmt.Printf("\n")
	points := make(plotter.XYs, 0, 8)
	for i := 0; i < Size; i++ {
		fmt.Println(proj.At(i, 0), proj.At(i, 1))
		points = append(points, plotter.XY{X: proj.At(i, 0), Y: proj.At(i, 1)})
	}

	p := plot.New()

	p.Title.Text = "x vs y"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(3)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "vectors.png")
	if err != nil {
		panic(err)
	}

	output, err := os.Create("vectors.dat")
	if err != nil {
		panic(err)
	}
	defer output.Close()
	for _, point := range points {
		fmt.Fprintf(output, "%f %f\n", point.X, point.Y)
	}
}
