--[[

	--------------------------------------------------------------------

	Version 0.4.0

	Aqwam's Tensor Library (TensorL)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	By using or possesing any copies of this library, you agree to our terms and conditions at:
	
	https://github.com/AqwamCreates/TensorL/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = {}

local function checkIfItHasSameDimensionSizeArray(dimensionSizeArray, targetDimensionSizeArray)

	if (#dimensionSizeArray ~= #targetDimensionSizeArray) then return false end

	for i, size in ipairs(dimensionSizeArray) do

		if (size ~= targetDimensionSizeArray[i]) then return false end

	end

	return true

end

local function checkIfDimensionIsOutOfBounds(dimension, minimumNumberOfDimensions, maximumNumberOfDimensions)

	if (dimension < minimumNumberOfDimensions) or  (dimension > maximumNumberOfDimensions) then error("The dimension is out of bounds.") end

end

local function checkIfDimensionSizeIndexIsOutOfBounds(dimensionSizeIndex, minimumDimensionSizeIndex, maximumDimensionSizeIndex)

	if (dimensionSizeIndex < minimumDimensionSizeIndex) or (dimensionSizeIndex > maximumDimensionSizeIndex) then error("The dimension size index is out of bounds.") end

end

local function removeFirstValueFromArray(array)

	local newArray = {}

	for i = 2, #array, 1 do table.insert(newArray, array[i]) end

	return newArray

end

local function removeLastValueFromArray(array)

	local newArray = table.clone(array)

	table.remove(newArray, #newArray)

	return newArray

end

local function createTensor(dimensionArray, initialValue)

	local result = {}

	if (#dimensionArray > 2) then

		local remainingDimensions = {}

		for i = 2, #dimensionArray do table.insert(remainingDimensions, dimensionArray[i]) end

		for i = 1, dimensionArray[1] do result[i] = createTensor(remainingDimensions, initialValue) end

	else

		for i = 1, dimensionArray[1] do result[i] = table.create(dimensionArray[2], initialValue) end

	end

	return result

end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function getNumberOfDimensions(tensor)
	
	if (typeof(tensor) ~= "table") then return 0 end
	
	return getNumberOfDimensions(tensor[1]) + 1
	
end

local function getSize(tensor, sizeArray)

	if (type(tensor) ~= "table") then return end

	table.insert(sizeArray, #tensor)

	getSize(tensor[1], sizeArray)

end

function AqwamTensorLibrary:getSize(tensor)

	local dimensionSizeArray = {}

	getSize(tensor, dimensionSizeArray)

	return dimensionSizeArray

end

local function applyFunctionUsingOneTensor(functionToApply, tensor, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = applyFunctionUsingOneTensor(functionToApply, tensor[i], remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = functionToApply(tensor[i]) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		newTensor = functionToApply(tensor)

	end

	return newTensor

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = applyFunctionUsingTwoTensors(functionToApply, tensor1[i], tensor2[i], remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = functionToApply(tensor1[i], tensor2[i]) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		newTensor = functionToApply(tensor1, tensor2)

	end

	return newTensor

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor[i], remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = functionToApply(scalar, tensor[i]) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		newTensor = functionToApply(scalar, tensor)

	end

	return newTensor

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local newTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor[i], scalar, remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = functionToApply(tensor[i], scalar) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		newTensor = functionToApply(tensor, scalar)

	end

	return newTensor

end

local function applyFunctionOnMultipleTensors(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local tensor = tensorArray[1]

	if (numberOfTensors == 1) then 

		local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

		if (type(tensor) == "table") then

			return applyFunctionUsingOneTensor(functionToApply, tensor, dimensionSizeArray)

		else

			return functionToApply(tensor, dimensionSizeArray)

		end

	end

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensorArray[i]

		local isFirstValueATensor = (type(tensor) == "table")

		local isSecondValueATensor = (type(otherTensor) == "table")

		if (isFirstValueATensor) and (isSecondValueATensor) then

			tensor, otherTensor = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor, otherTensor)

			local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (not isFirstValueATensor) and (isSecondValueATensor) then

			local dimensionSizeArray = AqwamTensorLibrary:getSize(otherTensor)

			tensor = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (isFirstValueATensor) and (not isSecondValueATensor) then

			local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

			tensor = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)

		else

			tensor = functionToApply(tensor, otherTensor)

		end

	end

	return tensor

end

local function sumFromAllDimensions(tensor, dimensionSizeArray)

	local numberOfDimensions = #dimensionSizeArray

	local result = 0

	if (numberOfDimensions > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do result = result + sumFromAllDimensions(tensor[i], remainingDimensionSizeArray) end

	else

		for i = 1, dimensionSizeArray[1], 1 do result = result + tensor[i] end

	end

	return result

end

local function recursiveSubTensorSumAlongFirstDimension(tensor, dimensionSizeArray, targetTensor, targetDimensionIndexArray)

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions >= 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do

			local copiedTargetDimensionIndexArray = table.clone(targetDimensionIndexArray)

			table.insert(copiedTargetDimensionIndexArray, i)

			recursiveSubTensorSumAlongFirstDimension(tensor[i], remainingDimensionSizeArray, targetTensor, copiedTargetDimensionIndexArray)

		end

	else

		targetDimensionIndexArray[1] = 1 -- The target dimension only have a size of 1 for summing.

		local targetTensorValue = AqwamTensorLibrary:getValue(targetTensor, targetDimensionIndexArray)

		local value = targetTensorValue + tensor

		AqwamTensorLibrary:setValue(targetTensor, value, targetDimensionIndexArray)

	end	

end

local function subTensorSumAlongFirstDimension(tensor, dimensionSizeArray)

	local sumDimensionalSizeArray = table.clone(dimensionSizeArray)

	sumDimensionalSizeArray[1] = 1

	local sumTensor = createTensor(sumDimensionalSizeArray, 0)

	recursiveSubTensorSumAlongFirstDimension(tensor, dimensionSizeArray, sumTensor, {})

	return sumTensor

end

local function sumAlongOneDimension(tensor, dimensionSizeArray, targetDimension, currentDimension)

	local newTensor

	if (currentDimension == targetDimension) then

		newTensor = subTensorSumAlongFirstDimension(tensor, dimensionSizeArray)

	else

		newTensor = {}

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = sumAlongOneDimension(tensor[i], remainingDimensionSizeArray, targetDimension, currentDimension + 1) end

	end

	return newTensor

end

function AqwamTensorLibrary:sum(dimension)
	
	dimension = dimension or 0

	local dimensionSizeArray = AqwamTensorLibrary:getSize(self)

	local numberOfDimensions = #dimensionSizeArray

	if (dimension == 0) then return sumFromAllDimensions(self, dimensionSizeArray) end

	checkIfDimensionIsOutOfBounds(dimension, 1, numberOfDimensions)

	local sumTensor = sumAlongOneDimension(self, dimensionSizeArray, dimension, 1)

	return self.new(sumTensor)
	
end

local function tensorProduct(tensor1, tensor2)
	
	local dimensionArray1 = getSize(tensor1)
	
	local dimensionArray2 = getSize(tensor2)

	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]
	
	local result = {}
	
	for i = 1, numberOfValues, 1 do

		if (#dimensionArray1 > 1) then

			local subproduct = tensorProduct(tensor1[i], tensor2[i])

			table.insert(result, subproduct)

		else

			table.insert(result, tensor1[i] * tensor2[i])

		end

	end

	return result
end

local function innerProduct(tensor1, tensor2)
	
	local dimensionArray1 = getSize(tensor1)

	local dimensionArray2 = getSize(tensor2)

	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end
	
	local numberOfValues = dimensionArray1[1]
	
	local result = 0
	
	for i = 1, numberOfValues, 1 do  
		
		if (#dimensionArray1 > 1) then

			result += innerProduct(tensor1[i], tensor2[i])

		else

			result += (tensor1[i] * tensor2[i])

		end
		
	end

	return result
	
end

local function outerProduct(tensor1, tensor2)
	
	local dimensionArray1 = getSize(tensor1)
	
	local dimensionArray2 = getSize(tensor2)

	for i, _ in ipairs(dimensionArray1) do if dimensionArray1[i] ~= dimensionArray2[i] then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]
	
	local result = {}
	
	for i = 1, numberOfValues do

		if (#dimensionArray1 > 1) then

			result[i] = outerProduct(tensor1[i], tensor2[i])

		else

			result[i] = {}

			for j = 1, numberOfValues do result[i][j] = tensor1[i] * tensor2[j] end

		end

	end

	return result
	
end

local function eq(booleanTensor)
	
	local dimensionArray1 = getSize(booleanTensor)

	local numberOfValues = dimensionArray1[1]

	local result = true

	if (#dimensionArray1 > 1) then

		for i = 1, numberOfValues do result = eq(booleanTensor[i]) end

	else

		for i = 1, numberOfValues do 
			
			result = (result == booleanTensor[i])
			
			if (result == false) then return false end
			
		end

	end

	return result
	
end

function AqwamTensorLibrary.new(...)
	
	local self = setmetatable({}, AqwamTensorLibrary)

	self.Values = ...

	return self
	
end

function AqwamTensorLibrary.createTensor(dimensionArray, initialValue)
	
	initialValue = initialValue or 0
	
	local self = setmetatable({}, AqwamTensorLibrary)
	
	self.Values = createTensor(dimensionArray, initialValue)
	
	return self
	
end

local function truncateDimensionSizeArrayIfRequired(dimensionSizeArray)

	local newDimensionSizeArray = table.clone(dimensionSizeArray)

	local numberOfStartingDimensionsWithTheSizeOf1 = 0

	while true do

		local size = newDimensionSizeArray[1]

		if (size ~= 1) then break end

		table.remove(newDimensionSizeArray, 1)

		numberOfStartingDimensionsWithTheSizeOf1 = numberOfStartingDimensionsWithTheSizeOf1 + 1

	end

	return newDimensionSizeArray, numberOfStartingDimensionsWithTheSizeOf1

end

local function createIdentityTensor(dimensionSizeArray, dimensionIndexArray)

	local numberOfDimensions = #dimensionSizeArray

	local tensor = {}

	if (numberOfDimensions >= 2) then

		for i = 1, dimensionSizeArray[1] do 

			local copiedDimensionIndexArray = table.clone(dimensionIndexArray)

			table.insert(copiedDimensionIndexArray, i)

			local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

			tensor[i] = createIdentityTensor(remainingDimensionSizeArray, copiedDimensionIndexArray) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do

			local copiedDimensionIndexArray = table.clone(dimensionIndexArray)

			local firstDimensionIndex = copiedDimensionIndexArray[1]

			table.insert(copiedDimensionIndexArray, i)

			tensor[i] = 1

			for _, dimensionIndex in ipairs(copiedDimensionIndexArray) do

				if (dimensionIndex ~= firstDimensionIndex) then

					tensor[i] = 0
					break

				end

			end

		end

	end

	return tensor

end

function AqwamTensorLibrary.createIdentityTensor(dimensionSizeArray)

	local truncatedDimensionSizeArray, numberOfDimensionsOfSize1 = truncateDimensionSizeArrayIfRequired(dimensionSizeArray)

	local newTensor = createIdentityTensor(truncatedDimensionSizeArray, {})

	for i = 1, numberOfDimensionsOfSize1, 1 do newTensor = {newTensor} end
	
	local self = setmetatable({}, AqwamTensorLibrary)
	
	self.Values = newTensor

	return self

end

local function createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	local tensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createRandomNormalTensor(remainingDimensionSizeArray, mean, standardDeviation) end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			local randomNumber1 = math.random()

			local randomNumber2 = math.random()

			local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2) -- Box–Muller transform formula.

			tensor[i] = (zScore * standardDeviation) + mean

		end

	end

	return tensor

end

function AqwamTensorLibrary.createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	mean = mean or 0

	standardDeviation = standardDeviation or 1
	
	local self = setmetatable({}, AqwamTensorLibrary)

	self.Values = createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	return self

end

local function createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	local numberOfDimensions = #dimensionSizeArray

	local tensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createRandomUniformTensor(remainingDimensionSizeArray, minimumValue, maximumValue) end

	elseif (numberOfDimensions == 1) and (minimumValue) and (maximumValue) then

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random(minimumValue, maximumValue) end

	elseif (numberOfDimensions == 1) and (minimumValue) and (not maximumValue) then

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random(minimumValue) end

	elseif (numberOfDimensions == 1) and (not minimumValue) and (not maximumValue) then

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random() end

	elseif (numberOfDimensions == 1) and (not minimumValue) and (maximumValue) then

		error("Invalid minimum value.")

	else

		error("An unknown error has occured when creating the random uniform tensor")

	end

	return tensor

end

function AqwamTensorLibrary.createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)
	
	local self = setmetatable({}, AqwamTensorLibrary)
	
	self.Values = createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	return self

end

function AqwamTensorLibrary:broadcast(dimensionsArray, values)

	local isNumber = typeof(values) == "number"

	if isNumber then return self.create(dimensionsArray, values) end
	
	return values

end

function AqwamTensorLibrary:getNumberOfDimensions()

	return getNumberOfDimensions(self)

end

function AqwamTensorLibrary:print()

	print(self)
	
end

local function hardcodedTranspose(tensor, targetDimensionArray) -- I don't think it is worth the effort to generalize to the rest of dimensions... That being said, to process videos, you need at most 5 dimensions. Don't get confused about the channels! Only number of channels are changed and not the number of dimensions of the tensor!

	local dimensionArray = AqwamTensorLibrary:getSize(tensor)

	local numberOfDimensions = #dimensionArray

	local offset = 5 - numberOfDimensions

	local dimensionSizeToAddArray = table.create(offset, 1)

	local expandedTensor = AqwamTensorLibrary:increaseNumberOfDimensions(tensor, dimensionSizeToAddArray)

	local targetDimension1 = targetDimensionArray[1] + offset
	local targetDimension2 = targetDimensionArray[2] + offset

	local expandedDimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor)

	targetDimensionArray = {targetDimension1, targetDimension2}

	expandedDimensionSizeArray[targetDimension1], expandedDimensionSizeArray[targetDimension2] = expandedDimensionSizeArray[targetDimension2], expandedDimensionSizeArray[targetDimension1]

	local newTensor = createTensor(expandedDimensionSizeArray, true)

	if (table.find(targetDimensionArray, 1)) and (table.find(targetDimensionArray, 2)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[b][a][c][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 1)) and (table.find(targetDimensionArray, 3)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[c][b][a][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 2)) and (table.find(targetDimensionArray, 3)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][c][b][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 1)) and (table.find(targetDimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[d][b][c][a][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 1)) and (table.find(targetDimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[e][b][c][d][a]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 2)) and (table.find(targetDimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][d][c][b][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 2)) and (table.find(targetDimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][e][c][d][b]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 3)) and (table.find(targetDimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][d][c][e]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 3)) and (table.find(targetDimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][e][d][c]

						end

					end

				end

			end

		end

	elseif (table.find(targetDimensionArray, 4)) and (table.find(targetDimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							newTensor[a][b][c][d][e] = expandedTensor[a][b][c][e][d]

						end

					end

				end

			end

		end

	else

		error("Invalid dimensions!")

	end

	return AqwamTensorLibrary:truncate(newTensor, offset)

end

function AqwamTensorLibrary:transpose(tensor, dimensionArray)

	if (AqwamTensorLibrary:getNumberOfDimensions(tensor) == 0) then return tensor end

	if (#dimensionArray ~= 2) then error("Dimension array must contain exactly 2 dimensions.") end

	if (dimensionArray[1] == dimensionArray[2]) then return tensor end
	
	local transposedTensor = hardcodedTranspose(self, dimensionArray)

	return self.new(transposedTensor)

end


function AqwamTensorLibrary:__eq(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a == b) end, self, other)
	
	local isEqual = eq(result)

	return isEqual
	
end

function AqwamTensorLibrary:isEqualTo(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a == b) end, self, other)

	return self.new(result)

end

function AqwamTensorLibrary:isGreaterThan(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a > b) end, self, other)

	return self.new(result)

end

function AqwamTensorLibrary:isGreaterOrEqualTo(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a >= b) end, self, other)

	return self.new(result)

end

function AqwamTensorLibrary:isLessThan(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a < b) end, self, other)
	
	return self.new(result)

end

function AqwamTensorLibrary:isLessOrEqualTo(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a <= b) end, self, other)

	return self.new(result)

end

function AqwamTensorLibrary:tensorProduct(other)
	
	local result = tensorProduct(self, other)
	
	return self.new(result)
	
end

function AqwamTensorLibrary:innerProduct(other)

	return innerProduct(self, other)

end

function AqwamTensorLibrary:outerProduct(other)
	
	local result = outerProduct(self, other)

	return self.new(result)

end

function AqwamTensorLibrary:copy()
	
	return deepCopyTable(self)
	
end

function AqwamTensorLibrary:rawCopy()
	
	return deepCopyTable(self.Values)
	
end

function AqwamTensorLibrary:__add(other)
	
	local result = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:add(other)
	
	local result = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__sub(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:subtract(other)
	
	local result = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__mul(...)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, ...)

	return self.new(result)
	
end

function AqwamTensorLibrary:multiply(...)
	
	local result = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, ...)

	return self.new(result)
	
end

function AqwamTensorLibrary:__div(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:divide(...)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, ...)

	return self.new(result)

end

function AqwamTensorLibrary:__unm()

	local result = applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

	return self.new(result)
	
end

function AqwamTensorLibrary:unaryMinus()

	local result = applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

	return self.new(result)

end

function AqwamTensorLibrary:__tostring()

	return self:generateTensorString()
	
end

function AqwamTensorLibrary:__len()
	
	return #self.Values
	
end

function AqwamTensorLibrary:__index(index)
	
	if (typeof(index) == "number") then
		
		return rawget(self.Values, index)
		
	else
		
		return rawget(AqwamTensorLibrary, index)
		
	end
	
end

function AqwamTensorLibrary:__newindex(index, value)
	
	rawset(self, index, value)
	
end

local function getOutOfBoundsIndexArray(array, arrayToBeCheckedForOutOfBounds)

	local outOfBoundsIndexArray = {}

	for i, value in ipairs(arrayToBeCheckedForOutOfBounds) do

		if (value < 1) or (value > array[i]) then table.insert(outOfBoundsIndexArray, i) end

	end

	return outOfBoundsIndexArray

end

local function extract(tensor, dimensionSizeArray, originDimensionIndexArray, targetDimensionIndexArray)

	local numberOfDimensions = #dimensionSizeArray

	local extractedTensor = {}

	local originDimensionIndex = originDimensionIndexArray[1]

	local targetDimensionIndex = targetDimensionIndexArray[1]

	if (numberOfDimensions >= 2) and (originDimensionIndex <= targetDimensionIndex) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingOriginDimensionIndexArray = removeFirstValueFromArray(originDimensionIndexArray)

		local remainingTargetDimensionIndexArray = removeFirstValueFromArray(targetDimensionIndexArray)

		for i = originDimensionIndex, targetDimensionIndex, 1 do 

			local extractedSubTensor = extract(tensor[i], remainingDimensionSizeArray, remainingOriginDimensionIndexArray, remainingTargetDimensionIndexArray) 

			table.insert(extractedTensor, extractedSubTensor)

		end

	elseif (numberOfDimensions >= 2) and (originDimensionIndex > targetDimensionIndex) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingOriginDimensionIndexArray = removeFirstValueFromArray(originDimensionIndexArray)

		local remainingTargetDimensionIndexArray = removeFirstValueFromArray(targetDimensionIndexArray)

		for i = targetDimensionIndex, #tensor, 1 do 

			local extractedSubTensor = extract(tensor[i], remainingDimensionSizeArray, remainingOriginDimensionIndexArray, remainingTargetDimensionIndexArray) 

			table.insert(extractedTensor, extractedSubTensor)

		end

		for i = 1, originDimensionIndex, 1 do 

			local extractedSubTensor = extract(tensor[i], remainingDimensionSizeArray, remainingOriginDimensionIndexArray, remainingTargetDimensionIndexArray) 

			table.insert(extractedTensor, extractedSubTensor)

		end

	elseif (numberOfDimensions == 1) and (originDimensionIndex <= targetDimensionIndex) then

		for i = originDimensionIndex, targetDimensionIndex, 1 do table.insert(extractedTensor, tensor[i]) end

	elseif (numberOfDimensions == 1) and (originDimensionIndex > targetDimensionIndex) then

		for i = targetDimensionIndex, #tensor, 1 do table.insert(extractedTensor, tensor[i]) end

		for i = 1, originDimensionIndex, 1 do table.insert(extractedTensor, tensor[i]) end
		
	else

		error("An unknown error has occured while extracting the tensor.")

	end

	return extractedTensor

end

function AqwamTensorLibrary:extract(originDimensionIndexArray, targetDimensionIndexArray)

	local dimensionSizeArray = AqwamTensorLibrary:getSize(self)

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions ~= #originDimensionIndexArray) then error("Invalid origin dimension index array.") end

	if (numberOfDimensions ~= #targetDimensionIndexArray) then error("Invalid target dimension index array.") end

	local outOfBoundsOriginIndexArray = getOutOfBoundsIndexArray(dimensionSizeArray, originDimensionIndexArray)

	local outOfBoundsTargetIndexArray = getOutOfBoundsIndexArray(dimensionSizeArray, targetDimensionIndexArray)

	local outOfBoundsOriginIndexArraySize = #outOfBoundsOriginIndexArray

	local outOfBoundsTargetIndexArraySize = #outOfBoundsTargetIndexArray

	if (outOfBoundsOriginIndexArraySize > 0) then

		local errorString = "Attempting to set an origin dimension index that is out of bounds for dimension at "

		for i, index in ipairs(outOfBoundsOriginIndexArray) do

			errorString = errorString .. index

			if (i < outOfBoundsOriginIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end

	if (outOfBoundsTargetIndexArraySize > 0) then

		local errorString = "Attempting to set an target dimension index that is out of bounds for dimension at "

		for i, index in ipairs(outOfBoundsTargetIndexArray) do

			errorString = errorString .. index

			if (i < outOfBoundsTargetIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end

	local extractedTensor = extract(self, dimensionSizeArray, originDimensionIndexArray, targetDimensionIndexArray)

	return self.new(extractedTensor)

end

local function dotProduct(tensor1, tensor2, tensor1DimensionSizeArray, tensor2DimensionSizeArray) -- Best one. Do not delete!

	local numberOfDimensions1 = #tensor1DimensionSizeArray

	local numberOfDimensions2 = #tensor2DimensionSizeArray

	local tensor = {}

	if (numberOfDimensions1 == 1) and (numberOfDimensions2 == 2) then

		for i = 1, #tensor1, 1 do -- Last dimension, so represents columns.

			tensor[i] = 0

			for j = 1, #tensor2[1], 1 do tensor[i] = (tensor1[i] * tensor2[i][j]) end -- Since tensor 1 column size matches with tensor 2 row size, we can use column index from tensor 1.

		end

	elseif (numberOfDimensions1 == 2) and (numberOfDimensions2 == 2) then

		local tensor1Row = #tensor1

		local tensor1Column = #tensor1[1]

		local tensor2Column = #tensor2[1]

		for row = 1, tensor1Row, 1 do

			tensor[row] = {}

			for column = 1, tensor2Column, 1 do

				local sum = 0

				for i = 1, tensor1Column do sum = sum + (tensor1[row][i] * tensor2[i][column]) end

				tensor[row][column] = sum

			end

		end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 > 2) then

		local remainingTensor1DimensionSizeArray = removeFirstValueFromArray(tensor1DimensionSizeArray)

		local remainingTensor2DimensionSizeArray = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2[i], remainingTensor1DimensionSizeArray, remainingTensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 == 2) then

		local remainingTensor1DimensionSizeArray = removeFirstValueFromArray(tensor1DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1] do tensor = dotProduct(tensor1[i], tensor2, remainingTensor1DimensionSizeArray, tensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 == 1) and (numberOfDimensions2 > 2) then

		local remainingTensor2DimensionSizeArray = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor2DimensionSizeArray[1] do tensor = dotProduct(tensor1, tensor2[i], tensor1DimensionSizeArray, remainingTensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 == 1) then

		for i = 1, tensor1DimensionSizeArray[1], 1 do

			for j = 1, tensor1DimensionSizeArray[2], 1 do 

				tensor[i] = {}

				local sum = 0

				for k = 1, tensor2DimensionSizeArray[1] do

					sum = sum + (tensor1[i][j] * tensor2[k]) 

				end

				tensor[i][j] = sum

			end

		end

	elseif (numberOfDimensions1 == 0) or (numberOfDimensions2 == 0) then

		tensor = AqwamTensorLibrary:multiply(tensor1, tensor2)

	else

		error({numberOfDimensions1, numberOfDimensions2})

	end

	return tensor

end

local function tensor2DimensionalDotProduct(tensor1, tensor2)

	local subTensor = {}

	local tensor1Row = #tensor1

	local tensor1Column = #tensor1[1]

	local tensor2Row = #tensor2

	local tensor2Column = #tensor2[1]

	if (tensor1Column ~= tensor2Row) then error("Unable to perform the dot product. The size of second last dimension of the first tensor does not equal to the size of the last dimension of the second tensor.") end

	for row = 1, tensor1Row, 1 do

		subTensor[row] = {}

		for column = 1, tensor2Column, 1 do

			local sum = 0

			for i = 1, tensor1Column do sum = sum + (tensor1[row][i] * tensor2[i][column]) end

			subTensor[row][column] = sum

		end

	end

	return subTensor

end

local function recursiveExpandedDotProduct(tensor1, tensor2, dimensionSizeArray1, dimensionSizeArray2) -- Since both have equal number of dimensions now, we only need to use only one dimension size array.

	local numberOfDimensions1 = #dimensionSizeArray1

	local numberOfDimensions2 = #dimensionSizeArray2

	local tensor

	if (numberOfDimensions1 >= 3) and (numberOfDimensions2 >= 3) and (dimensionSizeArray1[1] == dimensionSizeArray2[1]) then

		tensor = {}

		local remainingDimensionSizeArray1 = removeFirstValueFromArray(dimensionSizeArray1)

		local remainingDimensionSizeArray2 = removeFirstValueFromArray(dimensionSizeArray2)

		for i = 1, dimensionSizeArray1[1], 1 do tensor[i] = recursiveExpandedDotProduct(tensor1[i], tensor2[i], remainingDimensionSizeArray1, remainingDimensionSizeArray2) end

	elseif (numberOfDimensions1 == 2) and (numberOfDimensions2 == 2) and (dimensionSizeArray1[2] == dimensionSizeArray2[1]) then -- No need an elseif statement where number of dimension is 1. This operation requires 2D tensors.

		tensor = tensor2DimensionalDotProduct(tensor1, tensor2)

	elseif (numberOfDimensions1 == 0) or (numberOfDimensions2 == 0) then

		tensor = AqwamTensorLibrary:multiply(tensor1, tensor2)

	elseif (numberOfDimensions1 >= 2) and (numberOfDimensions2 >= 2) and (dimensionSizeArray1[1] ~= dimensionSizeArray2[1]) then

		error("Unable to dot product. The starting dimension sizes of the first tensor does not equal to the starting dimension sizes of the second tensor.")

	else

		error("Unable to dot product.")

	end

	return tensor

end

local function expandedDotProduct(tensor1, tensor2)

	local dimensionSizeArray1 =  AqwamTensorLibrary:getSize(tensor1)

	local dimensionSizeArray2 =  AqwamTensorLibrary:getSize(tensor2)

	local numberOfDimensions1 = #dimensionSizeArray1

	local numberOfDimensions2 = #dimensionSizeArray2

	local highestNumberOfDimensions = math.max(numberOfDimensions1, numberOfDimensions2)

	local numberOfDimensionsOffset1 = highestNumberOfDimensions - numberOfDimensions1

	local numberOfDimensionsOffset2 = highestNumberOfDimensions - numberOfDimensions2

	local expandedTensor1

	local expandedTensor2

	if (numberOfDimensionsOffset1 ~= 0) then

		local dimensionSizeToAddArray = {}

		for i = 1, numberOfDimensionsOffset1, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArray2[i]) end

		expandedTensor1 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor1, dimensionSizeToAddArray)

	else

		expandedTensor1 = tensor1

	end

	if (numberOfDimensionsOffset2 ~= 0) then

		local dimensionSizeToAddArray = {}

		for i = 1, numberOfDimensionsOffset2, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArray1[i]) end

		expandedTensor2 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor2, dimensionSizeToAddArray)

	else

		expandedTensor2 = tensor2

	end

	local expandedTensor1DimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor1)

	local expandedTensor2DimensionSizeArray = AqwamTensorLibrary:getSize(expandedTensor2)

	return recursiveExpandedDotProduct(expandedTensor1, expandedTensor2, expandedTensor1DimensionSizeArray, expandedTensor2DimensionSizeArray)

end

local function hardcodedDotProduct(tensor1, tensor2)

	local numberOfDimensions1 = AqwamTensorLibrary:getNumberOfDimensions(tensor1)

	local numberOfDimensions2 = AqwamTensorLibrary:getNumberOfDimensions(tensor2)

	local numberOfDimensionsOffset1 = 5 - numberOfDimensions1

	local numberOfDimensionsOffset2 = 5 - numberOfDimensions2

	local expandedTensor1 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor1, table.create(numberOfDimensionsOffset1, 1))

	local expandedTensor2 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor2, table.create(numberOfDimensionsOffset2, 1))

	local expandedNumberOfDimension1 = AqwamTensorLibrary:getSize(expandedTensor1)

	local expandedNumberOfDimension2 = AqwamTensorLibrary:getSize(expandedTensor2)

	local tensor = {}

	for a = 1, expandedNumberOfDimension1[1], 1 do

		tensor[a] = {}

		for b = 1, expandedNumberOfDimension1[2], 1 do

			tensor[a][b] = {}

			for c = 1, expandedNumberOfDimension1[3], 1 do

				tensor[a][b][c] = {}

				for d = 1, expandedNumberOfDimension1[4], 1 do

					tensor[a][b][c][d] = {}

					for e = 1, expandedNumberOfDimension2[5], 1 do

						tensor[a][b][c][d][e] = {}

						local sum = 0

						for f = 1, expandedNumberOfDimension1[5] do sum = sum + (expandedTensor1[a][b][c][d][f] * expandedTensor2[a][b][c][f][e]) end

						tensor[a][b][c][d][e] = sum

					end

				end

			end

		end

	end

	return tensor

end

function AqwamTensorLibrary:dotProduct(other) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	local result = expandedDotProduct(self, other)

	return self.new(result)

end

local function get2DTensorTextSpacing(tensor, dimensionSizeArray, textSpacingArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do textSpacingArray = get2DTensorTextSpacing(tensor[i], remainingDimensionSizeArray, textSpacingArray) end

	else

		for i = 1, dimensionSizeArray[1], 1 do textSpacingArray[i] = math.max(textSpacingArray[i], string.len(tostring(tensor[i]))) end

	end

	return textSpacingArray

end

function AqwamTensorLibrary:get2DTensorTextSpacing()

	local dimensionSizeArray = AqwamTensorLibrary:getSize(self)

	local numberOfDimensions = #dimensionSizeArray

	local sizeAtFinalDimension = dimensionSizeArray[numberOfDimensions]

	local textSpacingArray = table.create(sizeAtFinalDimension, 0)

	return get2DTensorTextSpacing(self, dimensionSizeArray, textSpacingArray)

end

local function generateTensorString(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generateTensorString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, tensorLength do

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. " "

		end

		text = text .. " }"

	end

	return text

end

function AqwamTensorLibrary:generateTensorString()

	local textSpacingArray = self:get2DTensorTextSpacing()

	local dimensionSizeArray = AqwamTensorLibrary:getSize(self)

	return generateTensorString(self, dimensionSizeArray, textSpacingArray, 1)

end

local function generateTensorStringWithComma(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generateTensorStringWithComma(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, tensorLength do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. ", "

		end

		text = text .. " }"

	end

	return text

end

function AqwamTensorLibrary:generateTensorStringWithComma()

	local textSpacingArray = self:get2DTensorTextSpacing()

	local dimensionSizeArray = self:getSize()

	return generateTensorString(self, dimensionSizeArray, textSpacingArray, 1)

end

local function generatePortableTensorString(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generatePortableTensorString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

		if (dimensionDepth > 1) then text = text .. "," end

	else

		text = text .. "{ "

		for i = 1, tensorLength do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. ", "

		end

		text = text .. " },"

	end

	return text

end

function AqwamTensorLibrary:generatePortableTensorString()

	local textSpacingArray = self:get2DTensorTextSpacing()

	local dimensionSizeArray = self:getSize()

	return generatePortableTensorString(self, dimensionSizeArray, textSpacingArray, 1)

end

function AqwamTensorLibrary:printTensor(tensor)

	print("\n\n" .. self:generateTensorString() .. "\n\n")

end

function AqwamTensorLibrary:printTensorWithComma()

	print("\n\n" .. self:generateTensorStringWithComma() .. "\n\n")

end

function AqwamTensorLibrary:printPortableTensor()

	print("\n\n" .. self:generatePortableTensorString() .. "\n\n")

end

local function expand(tensor, dimensionSizeArray, targetDimensionSizeArray)

	-- Does not do the same thing with inefficient expand function. This one expand at the lowest dimension first and then the parent dimension will make copy of this.

	local newTensor

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions >= 2) then

		newTensor = {}

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingTargetDimensionSizeArray = removeFirstValueFromArray(targetDimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do newTensor[i] = expand(tensor[i], remainingDimensionSizeArray, remainingTargetDimensionSizeArray) end

	else

		newTensor = deepCopyTable(tensor)  -- If the "(numberOfDimensions > 1)" from the first "if" statement does not run, it will return the original tensor. So we need to deep copy it.

	end

	local updatedDimensionSizeArray = AqwamTensorLibrary:getSize(newTensor) -- Need to call this again because we may have modified the tensor below it, thus changing the dimension size array.

	local dimensionSize = updatedDimensionSizeArray[1]

	local targetDimensionSize = targetDimensionSizeArray[1]

	local hasSameDimensionSize = (dimensionSize == targetDimensionSize)

	local canDimensionBeExpanded = (dimensionSize == 1)

	if (numberOfDimensions >= 1) and (not hasSameDimensionSize) and (canDimensionBeExpanded) then 

		local subTensor = newTensor[1]

		for i = 1, targetDimensionSize, 1 do newTensor[i] = deepCopyTable(subTensor) end

	elseif (not hasSameDimensionSize) and (not canDimensionBeExpanded) then

		error("Unable to expand.")

	end

	return newTensor

end

function AqwamTensorLibrary:expand(targetDimensionSizeArray)

	local dimensionSizeArray = self:getSize()

	if checkIfItHasSameDimensionSizeArray(dimensionSizeArray, targetDimensionSizeArray) then return deepCopyTable(tensor) end -- Do not remove this code even if the code below is related or function similar to this code. You will spend so much time fixing it if you forget that you have removed it.
	
	local newTensor = expand(self, dimensionSizeArray, targetDimensionSizeArray)
	
	return self.new(newTensor)

end

function AqwamTensorLibrary:increaseNumberOfDimensions(dimensionSizeToAddArray)

	local newTensor = {}

	local numberOfDimensionsToAdd = #dimensionSizeToAddArray

	if (numberOfDimensionsToAdd > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeToAddArray)

		for i = 1, dimensionSizeToAddArray[1], 1 do newTensor[i] = self:increaseNumberOfDimensions(remainingDimensionSizeArray) end

	elseif (numberOfDimensionsToAdd == 1) then

		for i = 1, dimensionSizeToAddArray[1], 1 do newTensor[i] = deepCopyTable(self) end

	else

		newTensor = self.Values

	end
	
	if (#dimensionSizeToAddArray == #self:getSize()) then
		
		return self.new(newTensor)
			
	else
		
		return newTensor
		
	end

end

return AqwamTensorLibrary
