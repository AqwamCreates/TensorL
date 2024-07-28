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

local function getSubTensorLength(tensor, targetDimension)
	
	local numberOfDimensions = getNumberOfDimensions(tensor)
	
	if (numberOfDimensions == targetDimension) then return #tensor end
	
	return getSubTensorLength(tensor[1], targetDimension)
	
end

local function getDimensionArray(tensor)
	
	local numberOfDimensions = getNumberOfDimensions(tensor)

	local dimensionArray = {}

	for dimension = numberOfDimensions, 1, -1  do

		local length = getSubTensorLength(tensor, dimension)

		table.insert(dimensionArray, length)

	end
	
	return dimensionArray
	
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

		local isFirstTensorATensor = (type(tensor) == "table")

		local isSecondTensorATensor = (type(otherTensor) == "table")

		if (isFirstTensorATensor) and (isSecondTensorATensor) then

			tensor, otherTensor = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor, otherTensor)

			local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (isFirstTensorATensor) and (not isSecondTensorATensor) then

			local dimensionSizeArray = AqwamTensorLibrary:getSize(tensor)

			tensor = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (not isFirstTensorATensor) and (isSecondTensorATensor) then

			local dimensionSizeArray = AqwamTensorLibrary:getSize(otherTensor)

			tensor = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)

		else

			tensor = functionToApply(tensor, otherTensor)

		end

	end

	return tensor

end

local function createString(tensor)

	local dimensionArray = getDimensionArray(tensor)
	
	local tensorLength = #tensor

	local result = " "

	if (#dimensionArray > 1) then
		
		result = result .. "{"

		for i = 1, #tensor do 
			
			result = result .. createString(tensor[i])
			
			if (i == tensorLength) then continue end
			
			result = result .. "\n"
			
		end
		
		result = result .. " }"

	else

		result = result .. "{ "

		for i = 1, tensorLength do 

			result = result .. tensor[i]

			if (i == tensorLength) then continue end

			result = result .. ", "

		end

		result = result .. " }"

	end

	return result

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
	
	local dimensionArray1 = getDimensionArray(tensor1)
	
	local dimensionArray2 = getDimensionArray(tensor2)

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
	
	local dimensionArray1 = getDimensionArray(tensor1)

	local dimensionArray2 = getDimensionArray(tensor2)

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
	
	local dimensionArray1 = getDimensionArray(tensor1)
	
	local dimensionArray2 = getDimensionArray(tensor2)

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
	
	local dimensionArray1 = getDimensionArray(booleanTensor)

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

local function transpose(tensor, dimension1, dimension2)
	
	local dimensionArray = getDimensionArray(tensor)
	
	local numberOfDimensions = #dimensionArray
	
end

function AqwamTensorLibrary.new(...)
	
	local self = setmetatable({}, AqwamTensorLibrary)

	self.Values = ...

	return self
	
end

function AqwamTensorLibrary.create(dimensionArray, initialValue)
	
	initialValue = initialValue or 0
	
	local self = setmetatable({}, AqwamTensorLibrary)
	
	self.Values = createTensor(dimensionArray, initialValue)
	
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

function AqwamTensorLibrary:getDimensionArray()
	
	return getDimensionArray(self)
	
end

function AqwamTensorLibrary:print()

	print(self)
	
end

function AqwamTensorLibrary:transpose(dimension1, dimension2)
	
	if (typeof(dimension1) ~= "number") or (typeof(dimension2) ~= "number") then error("Dimensions are not numbers.") end
	
	local numberOfDimension = getNumberOfDimensions(self)

	if (dimension1 < 1) or (dimension1 > numberOfDimension) or (dimension2 < 1) or (dimension2 > numberOfDimension) or (dimension1 == dimension2) then error("Invalid dimensions.") end
	
	local result = transpose(self, dimension1, dimension2)

	return result
	
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

	return self.create(result)

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

function AqwamTensorLibrary:__sub(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__mul(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__div(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__unm(other)

	local result = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__tostring()
	
	local text = "\n\n" .. createString(self) .. "\n\n"

	return text
	
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

return AqwamTensorLibrary
